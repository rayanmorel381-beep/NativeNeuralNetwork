use crate::graph::blocks::embeddings::gather_embeddings;
use crate::graph::attn::kv_cache::KvCacheView;
use crate::graph::lm::{LmBlockWeights, LmConfig, LmError, LmWeights};
use crate::base::math::Float;
use crate::graph::blocks::normalization::rms_norm_in_place;
use crate::graph::attn::rope::{apply_rope_from_cache, precompute_rope_freqs};
use crate::engine::runtime::{axpy, dot, parallel_for, SyncMutPtr};
use crate::engine::runtime::hardware::{mmap_shared_anon, munmap};

fn prefill_matmul_rows<T: Float>(x: &[T], w: &[T], rows: usize, in_d: usize, out_d: usize, y: &mut [T]) {
    let y_ptr = SyncMutPtr(y.as_mut_ptr());
    parallel_for(out_d, in_d.saturating_mul(rows), &move |o| {
        let _ = &y_ptr;
        let wrow = &w[o * in_d..o * in_d + in_d];
        for r in 0..rows {
            let xr = &x[r * in_d..r * in_d + in_d];
            unsafe { *y_ptr.0.add(r * out_d + o) = dot(xr, wrow, in_d); }
        }
    });
}

#[inline]
fn silu<T: Float>(x: T) -> T {
    x / (T::ONE + (-x).exp())
}

pub fn prefill_buf_count<T: Float>(cfg: &LmConfig<T>, s: usize) -> usize {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    5 * s * h + 2 * s * kv_h + s + 2 * s * ffw
}

fn prefill_layer<T: Float>(
    cfg: &LmConfig<T>,
    bw: &LmBlockWeights<T>,
    kv: &mut KvCacheView<T>,
    hiddens: &mut [T],
    normed: &mut [T],
    q: &mut [T],
    k: &mut [T],
    vv: &mut [T],
    scores_buf: &mut [T],
    attn_out: &mut [T],
    gate: &mut [T],
    up: &mut [T],
    cos_cache: &[T],
    sin_cache: &[T],
    pairs: usize,
    s: usize,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let head_dim = cfg.head_dim()?;
    let groups = cfg.kv_head_groups();
    let scale = T::ONE / T::from_usize(head_dim).sqrt();
    let eps = T::from_f32(1e-5);

    for p in 0..s {
        normed[p * h..p * h + h].copy_from_slice(&hiddens[p * h..p * h + h]);
        rms_norm_in_place(&mut normed[p * h..p * h + h], bw.attn_norm_gamma, eps)
            .map_err(|_| LmError::ShapeMismatch)?;
    }

    prefill_matmul_rows(normed, bw.wq, s, h, h, q);
    prefill_matmul_rows(normed, bw.wk, s, h, kv_h, k);
    prefill_matmul_rows(normed, bw.wv, s, h, kv_h, vv);

    for p in 0..s {
        for head in 0..nh {
            let off = p * h + head * head_dim;
            apply_rope_from_cache(&mut q[off..off + head_dim], p, cos_cache, sin_cache, pairs)
                .map_err(|_| LmError::ShapeMismatch)?;
        }
        for kv_head in 0..nkv {
            let off = p * kv_h + kv_head * head_dim;
            apply_rope_from_cache(&mut k[off..off + head_dim], p, cos_cache, sin_cache, pairs)
                .map_err(|_| LmError::ShapeMismatch)?;
        }
    }

    for p in 0..s {
        kv.append_token(&k[p * kv_h..p * kv_h + kv_h], &vv[p * kv_h..p * kv_h + kv_h])
            .map_err(|_| LmError::KvCapacityExceeded)?;
    }

    for idx in 0..s * h {
        attn_out[idx] = T::ZERO;
    }

    for head in 0..nh {
        let kv_head = head / groups;
        for p in 0..s {
            let qh = &q[p * h + head * head_dim..p * h + (head + 1) * head_dim];
            let scores = &mut scores_buf[..p + 1];
            let mut max = T::NEG_INF;
            for j in 0..=p {
                let kj = &k[j * kv_h + kv_head * head_dim..j * kv_h + (kv_head + 1) * head_dim];
                let sc = dot(qh, kj, head_dim) * scale;
                scores[j] = sc;
                if sc > max {
                    max = sc;
                }
            }
            let mut sum = T::ZERO;
            for sc in scores.iter_mut() {
                *sc = (*sc - max).exp();
                sum += *sc;
            }
            if sum > T::ZERO {
                for sc in scores.iter_mut() {
                    *sc /= sum;
                }
            }
            let ao = &mut attn_out[p * h + head * head_dim..p * h + (head + 1) * head_dim];
            for j in 0..=p {
                let pw = scores[j];
                if pw.abs() > T::ZERO {
                    let vj = &vv[j * kv_h + kv_head * head_dim..j * kv_h + (kv_head + 1) * head_dim];
                    axpy(ao, pw, vj, head_dim);
                }
            }
        }
    }

    prefill_matmul_rows(attn_out, bw.wo, s, h, h, normed);
    for i in 0..s * h {
        hiddens[i] += normed[i];
    }

    for p in 0..s {
        normed[p * h..p * h + h].copy_from_slice(&hiddens[p * h..p * h + h]);
        rms_norm_in_place(&mut normed[p * h..p * h + h], bw.ffn_norm_gamma, eps)
            .map_err(|_| LmError::ShapeMismatch)?;
    }

    prefill_matmul_rows(normed, bw.w_gate, s, h, ffw, gate);
    prefill_matmul_rows(normed, bw.w_up, s, h, ffw, up);

    for idx in 0..s * ffw {
        gate[idx] = silu(gate[idx]) * up[idx];
    }

    prefill_matmul_rows(gate, bw.w_down, s, ffw, h, normed);
    for i in 0..s * h {
        hiddens[i] += normed[i];
    }

    Ok(())
}

fn prefill_inner<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    prompt_ids: &[u32],
    kv_layers: &mut [KvCacheView<T>],
    out_last_hidden: &mut [T],
    work: &mut [T],
    s: usize,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    let v = cfg.vocab_size;

    if kv_layers.len() < cfg.num_layers {
        return Err(LmError::ShapeMismatch);
    }

    let (hiddens, rest) = work.split_at_mut(s * h);
    let (normed, rest) = rest.split_at_mut(s * h);
    let (q, rest) = rest.split_at_mut(s * h);
    let (k, rest) = rest.split_at_mut(s * kv_h);
    let (vv, rest) = rest.split_at_mut(s * kv_h);
    let (scores_buf, rest) = rest.split_at_mut(s);
    let (attn_out, rest) = rest.split_at_mut(s * h);
    let (gate, rest) = rest.split_at_mut(s * ffw);
    let (up, rope_rest) = rest.split_at_mut(s * ffw);

    let head_dim = if cfg.num_heads > 0 { h / cfg.num_heads } else { 0 };
    let pairs = head_dim / 2;
    let (cos_cache, sin_cache) = if pairs > 0 && rope_rest.len() >= 2 * s * pairs {
        rope_rest.split_at_mut(s * pairs)
    } else {
        rope_rest.split_at_mut(0)
    };
    if pairs > 0 && cos_cache.len() >= s * pairs && sin_cache.len() >= s * pairs {
        precompute_rope_freqs(cos_cache, sin_cache, s, head_dim, cfg.rope_theta)
            .map_err(|_| LmError::ShapeMismatch)?;
    }

    for (i, &tid) in prompt_ids.iter().enumerate() {
        gather_embeddings(w.embed_table, v, h, &[tid as usize], &mut hiddens[i * h..(i + 1) * h])
            .map_err(|_| LmError::ShapeMismatch)?;
    }

    for (block, kv) in w.blocks.iter().zip(kv_layers.iter_mut()) {
        prefill_layer(cfg, block, kv, hiddens, normed, q, k, vv, scores_buf, attn_out, gate, up, cos_cache, sin_cache, pairs, s)?;
    }

    rms_norm_in_place(&mut hiddens[(s - 1) * h..s * h], w.final_norm_gamma, T::from_f32(1e-5))
        .map_err(|_| LmError::ShapeMismatch)?;

    if out_last_hidden.len() >= h {
        out_last_hidden[..h].copy_from_slice(&hiddens[(s - 1) * h..s * h]);
    }
    Ok(())
}

pub fn prefill_prompt<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    prompt_ids: &[u32],
    kv_layers: &mut [KvCacheView<T>],
    out_last_hidden: &mut [T],
) -> Result<(), LmError> {
    let s = prompt_ids.len();
    if s == 0 {
        return Err(LmError::ShapeMismatch);
    }
    let elem = core::mem::size_of::<T>();
    let work_count = prefill_buf_count(cfg, s);
    let work_bytes = work_count * elem;
    let ptr = mmap_shared_anon(work_bytes);
    if ptr.is_null() {
        return Err(LmError::ScratchTooSmall);
    }
    let work = unsafe { core::slice::from_raw_parts_mut(ptr as *mut T, work_count) };
    let result = prefill_inner(cfg, w, prompt_ids, kv_layers, out_last_hidden, work, s);
    munmap(ptr, work_bytes);
    result
}
