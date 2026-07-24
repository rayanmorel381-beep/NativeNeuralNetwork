use crate::graph::attn::kv_cache::KvCacheView;
use crate::graph::lm::{LmBlockWeights, LmConfig, LmError};
use crate::graph::lm::LmBlockGrads;
use crate::base::math::Float;
use crate::base::tensor::TensorViewMut;
use super::activations::BlockActivations;
use super::transformer_block::silu_prime;

pub struct BackwardScratch<'a, T> {
    pub tmp1: &'a mut [T],
    pub tmp2: &'a mut [T],
    pub d_scores: &'a mut [T],
}

pub(crate) fn rmsnorm_backward<T: Float>(
    pre_norm: &[T],
    gamma: &[T],
    d_out: &[T],
    d_gamma: &mut [T],
    d_x: &mut [T],
    h: usize,
) {
    if TensorViewMut::<T>::from_slice_offset(d_x, [1, 1, 1, 1, h], 0).len() < h {
        return;
    }
    if h == 0 {
        TensorViewMut::from_slice(d_gamma, [1, 1, 1, 1, 0]).fill(T::ZERO);
        return;
    }
    let mut sumsq = T::ZERO;
    for xi in &pre_norm[..h] {
        sumsq += *xi * *xi;
    }
    let rms_sq = sumsq / T::from_usize(h) + T::from_f32(1e-5);
    let inv_rms = T::ONE / rms_sq.sqrt();
    let mut dot = T::ZERO;
    for i in 0..h {
        dot += gamma[i] * d_out[i] * pre_norm[i] * inv_rms;
    }
    let mean_dot = dot / T::from_usize(h);
    for i in 0..h {
        let xhat = pre_norm[i] * inv_rms;
        d_gamma[i] += xhat * d_out[i];
        d_x[i] += inv_rms * (gamma[i] * d_out[i] - xhat * mean_dot);
    }
}

fn outer_accum<T: Float>(dy: &[T], x: &[T], out: &mut [T], out_d: usize, in_d: usize) {
    let mut tv = TensorViewMut::from_slice(out, [1, 1, 1, out_d, in_d]);
    if tv.is_empty() { return; }
    let out_d = tv.shape[3];
    let in_d = if out_d > 0 { tv.len() / out_d } else { return; };
    if tv.offset != 0 { return; }
    if out_d == 1 {
        if !dy.is_empty() { tv.axpy(dy[0], x); }
        return;
    }
    let out_ptr = crate::engine::runtime::SyncMutPtr(tv.data.as_mut_ptr());
    crate::engine::runtime::parallel_for(out_d, in_d, &move |o| {
        let _ = &out_ptr;
        let base = o * in_d;
        let row = unsafe { core::slice::from_raw_parts_mut(out_ptr.0.add(base), in_d) };
        crate::engine::runtime::axpy(row, dy[o], &x[..in_d], in_d);
    });
}

fn matvec_t_accum<T: Float>(dy: &[T], w: &[T], out: &mut [T], out_d: usize, in_d: usize) {
    let tile = {
        let l2 = crate::engine::runtime::l2_cache_bytes();
        if l2 >= 512 * 1024 { 128 }
        else if l2 >= 256 * 1024 { 64 }
        else if l2 > 0 { 32 }
        else { 64 }
    };
    let n_tiles = in_d.div_ceil(tile);
    let out_ptr = crate::engine::runtime::SyncMutPtr(out.as_mut_ptr());
    crate::engine::runtime::parallel_for(n_tiles, out_d.saturating_mul(tile), &move |t| {
        let _ = &out_ptr;
        let i0 = t * tile;
        let i1 = (i0 + tile).min(in_d);
        let width = i1 - i0;
        let dst = unsafe { core::slice::from_raw_parts_mut(out_ptr.0.add(i0), width) };
        for o in 0..out_d {
            let a = dy[o];
            if a.abs() > T::ZERO {
                let row = &w[o * in_d + i0..o * in_d + i1];
                crate::engine::runtime::axpy(dst, a, row, width);
            }
        }
    });
}

pub fn transformer_block_backward<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmBlockWeights<T>,
    acts: &BlockActivations<T>,
    kv: &KvCacheView<T>,
    d_out: &mut [T],
    grads: &mut LmBlockGrads<T>,
    sc: &mut BackwardScratch<T>,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim()?;
    let nh = cfg.num_heads;
    let kv_h = cfg.kv_h();
    let groups = cfg.kv_head_groups();
    let ffw = cfg.ffw_size;
    let seq_len = acts.seq_len;

    let req_scores = seq_len + if h > 2 * kv_h { h } else { 2 * kv_h };
    if d_out.len() < h
        || sc.tmp1.len() < h.max(ffw)
        || sc.tmp2.len() < h.max(ffw)
        || sc.d_scores.len() < req_scores
    {
        return Err(LmError::ScratchTooSmall);
    }

    for i in 0..ffw {
        let s = acts.gate_pre[i] / (T::ONE + (-acts.gate_pre[i]).exp());
        sc.tmp1[i] = s * acts.up[i];
    }
    outer_accum(&d_out[..h], &sc.tmp1[..ffw], grads.w_down, h, ffw);

    sc.tmp2[..ffw].iter_mut().for_each(|v| *v = T::ZERO);
    matvec_t_accum(&d_out[..h], w.w_down, &mut sc.tmp2[..ffw], h, ffw);

    for i in 0..ffw {
        let d_gs = sc.tmp2[i];
        let sp = silu_prime(acts.gate_pre[i]);
        let s = acts.gate_pre[i] / (T::ONE + (-acts.gate_pre[i]).exp());
        sc.tmp1[i] = d_gs * sp * acts.up[i];
        sc.tmp2[i] = d_gs * s;
    }
    outer_accum(&sc.tmp1[..ffw], &acts.post_norm_ffn[..h], grads.w_gate, ffw, h);
    outer_accum(&sc.tmp2[..ffw], &acts.post_norm_ffn[..h], grads.w_up, ffw, h);

    let pnf_off = seq_len;
    sc.d_scores[pnf_off..pnf_off + h].iter_mut().for_each(|v| *v = T::ZERO);
    {
        let (lo, _) = sc.d_scores.split_at_mut(pnf_off + h);
        matvec_t_accum(&sc.tmp1[..ffw], w.w_gate, &mut lo[pnf_off..], ffw, h);
        matvec_t_accum(&sc.tmp2[..ffw], w.w_up, &mut lo[pnf_off..], ffw, h);
    }

    sc.tmp1[..h].iter_mut().for_each(|v| *v = T::ZERO);
    rmsnorm_backward(
        &acts.pre_norm_ffn[..h],
        w.ffn_norm_gamma,
        &sc.d_scores[pnf_off..pnf_off + h],
        &mut grads.ffn_norm_gamma[..h],
        &mut sc.tmp1[..h],
        h,
    );
    for (d, &t) in d_out[..h].iter_mut().zip(sc.tmp1[..h].iter()) {
        *d += t;
    }

    outer_accum(&d_out[..h], &acts.pre_wo[..h], grads.wo, h, h);

    sc.tmp2[..h].iter_mut().for_each(|v| *v = T::ZERO);
    matvec_t_accum(&d_out[..h], w.wo, &mut sc.tmp2[..h], h, h);

    sc.tmp1[..h].iter_mut().for_each(|v| *v = T::ZERO);

    let dk_off = seq_len;
    let dv_off = seq_len + kv_h;
    sc.d_scores[dk_off..dk_off + kv_h].iter_mut().for_each(|v| *v = T::ZERO);
    sc.d_scores[dv_off..dv_off + kv_h].iter_mut().for_each(|v| *v = T::ZERO);

    let kv_stride = kv_h;
    let scale = T::ONE / T::from_usize(head_dim).sqrt();
    let cur_pos = seq_len.saturating_sub(1);

    for head in 0..nh {
        let q_off = head * head_dim;
        let kv_head = head / groups;
        let kv_head_off = kv_head * head_dim;
        let score_src = head * cfg.context_len;
        let scores = &acts.attn_scores_per_head[score_src..score_src + seq_len];

        let mut dot_s_ds = T::ZERO;
        for (t, (&score, d_score)) in scores.iter().zip(sc.d_scores[..seq_len].iter_mut()).enumerate() {
            let v_base = t * kv_stride + kv_head_off;
            let mut d_st = T::ZERO;
            for i in 0..head_dim {
                d_st += sc.tmp2[q_off + i] * kv.value[v_base + i];
            }
            *d_score = d_st;
            dot_s_ds += score * d_st;
        }
        for (d, &s) in sc.d_scores[..seq_len].iter_mut().zip(scores.iter()) {
            *d = s * (*d - dot_s_ds) * scale;
        }

        for t in 0..seq_len {
            let k_base = t * kv_stride + kv_head_off;
            let dps = sc.d_scores[t];
            for i in 0..head_dim {
                sc.tmp1[q_off + i] += dps * kv.key[k_base + i];
            }
        }

        let dps_cur = sc.d_scores[cur_pos];
        for i in 0..head_dim {
            sc.d_scores[dk_off + kv_head_off + i] += dps_cur * acts.q[q_off + i];
        }

        let s_cur = scores[cur_pos];
        for i in 0..head_dim {
            sc.d_scores[dv_off + kv_head_off + i] += s_cur * sc.tmp2[q_off + i];
        }
    }

    outer_accum(&sc.tmp1[..h], &acts.post_norm_attn[..h], grads.wq, h, h);
    outer_accum(&sc.d_scores[dk_off..dk_off + kv_h], &acts.post_norm_attn[..h], grads.wk, kv_h, h);
    outer_accum(&sc.d_scores[dv_off..dv_off + kv_h], &acts.post_norm_attn[..h], grads.wv, kv_h, h);

    sc.tmp2[..h].iter_mut().for_each(|v| *v = T::ZERO);
    matvec_t_accum(&sc.tmp1[..h], w.wq, &mut sc.tmp2[..h], h, h);
    matvec_t_accum(&sc.d_scores[dk_off..dk_off + kv_h], w.wk, &mut sc.tmp2[..h], kv_h, h);

    sc.tmp1[..h].iter_mut().for_each(|v| *v = T::ZERO);
    rmsnorm_backward(
        &acts.pre_norm_attn[..h],
        w.attn_norm_gamma,
        &sc.tmp2[..h],
        &mut grads.attn_norm_gamma[..h],
        &mut sc.tmp1[..h],
        h,
    );
    for (d, &t) in d_out[..h].iter_mut().zip(sc.tmp1[..h].iter()) {
        *d += t;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn mlp_head_backward<T: Float>(
    d_logits: &[T],
    hidden: &[T],
    a1: &[T],
    w1: &[T],
    w2: &[T],
    vocab_size: usize,
    hidden_size: usize,
    head_hidden: usize,
    d_w1: &mut [T],
    d_b1: &mut [T],
    d_w2: &mut [T],
    d_b2: &mut [T],
    d_z1: &mut [T],
    d_hidden: &mut [T],
) -> Result<(), LmError> {
    if d_logits.len() < vocab_size
        || hidden.len() < hidden_size
        || a1.len() < head_hidden
        || w1.len() < head_hidden * hidden_size
        || w2.len() < vocab_size * head_hidden
        || d_w1.len() < head_hidden * hidden_size
        || d_b1.len() < head_hidden
        || d_w2.len() < vocab_size * head_hidden
        || d_b2.len() < vocab_size
        || d_z1.len() < head_hidden
        || d_hidden.len() < hidden_size
    {
        return Err(LmError::ShapeMismatch);
    }

    let d_w2_ptr = crate::engine::runtime::SyncMutPtr(d_w2.as_mut_ptr());
    crate::engine::runtime::parallel_for(vocab_size, head_hidden, &move |k| {
        let _ = &d_w2_ptr;
        let dl = d_logits[k];
        let off = k * head_hidden;
        for (j, &aj) in a1[..head_hidden].iter().enumerate() {
            unsafe {
                *d_w2_ptr.0.add(off + j) += dl * aj;
            }
        }
    });
    for k in 0..vocab_size {
        d_b2[k] += d_logits[k];
    }

    for j in 0..head_hidden {
        let mut acc = T::ZERO;
        for k in 0..vocab_size {
            acc += d_logits[k] * w2[k * head_hidden + j];
        }
        let g = if a1[j] > T::ZERO { acc } else { T::ZERO };
        d_z1[j] = g;
        d_b1[j] += g;
    }

    let d_z1: &[T] = d_z1;
    let d_w1_ptr = crate::engine::runtime::SyncMutPtr(d_w1.as_mut_ptr());
    crate::engine::runtime::parallel_for(head_hidden, hidden_size, &move |j| {
        let _ = &d_w1_ptr;
        let dz = d_z1[j];
        let off = j * hidden_size;
        for (i, &hi) in hidden[..hidden_size].iter().enumerate() {
            unsafe {
                *d_w1_ptr.0.add(off + i) += dz * hi;
            }
        }
    });

    let d_hidden_ptr = crate::engine::runtime::SyncMutPtr(d_hidden.as_mut_ptr());
    crate::engine::runtime::parallel_for(hidden_size, head_hidden, &move |i| {
        let _ = &d_hidden_ptr;
        let mut acc = T::ZERO;
        for j in 0..head_hidden {
            acc += d_z1[j] * w1[j * hidden_size + i];
        }
        unsafe {
            *d_hidden_ptr.0.add(i) = acc;
        }
    });

    Ok(())
}
