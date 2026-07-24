use crate::graph::attn::attention::{sdpa_strided, sdpa_with_sparse_mask, StridedHead};
use crate::graph::attn::kv_cache::KvCacheView;
use crate::graph::blocks::normalization::rms_norm_in_place;
use crate::graph::attn::rope::apply_rope_in_place;
use crate::graph::lm::{LmBlockWeights, LmBlockLora, LmConfig, LmError, BlockQWeights, QWeight, BlockTWeights, TWeight};
use crate::graph::blocks::lora::lora_forward_delta;
use crate::format::quantization::{matvec_ternary, matvec_w8};
use crate::base::math::Float;
use super::scratch::ForwardScratch;
use super::activations::BlockActivations;

fn matmul<T: Float>(
    inp: &[T],
    w: &[T],
    qw: Option<QWeight<'_, T>>,
    tw: Option<TWeight<'_, T>>,
    in_d: usize,
    out_d: usize,
    out: &mut [T],
) -> Result<(), LmError> {
    if inp.len() < in_d || out.len() < out_d {
        return Err(LmError::ShapeMismatch);
    }
    if let Some(t) = tw {
        if t.scales.len() < out_d {
            return Err(LmError::ShapeMismatch);
        }
        matvec_ternary(&inp[..in_d], t.packed, t.scales, in_d, out_d, &mut out[..out_d])
            .map_err(|_| LmError::ShapeMismatch)?;
        return Ok(());
    }
    if let Some(q) = qw {
        if q.q.len() < out_d * in_d || q.scales.len() < out_d {
            return Err(LmError::ShapeMismatch);
        }
        let packed = crate::base::tensor::PackedTensor::from_raw(q.q, [1, 1, 1, out_d, in_d]);
        let quant = crate::base::tensor::QuantizedTensor::from_qweight(q.q, q.scales, in_d, [1, 1, 1, out_d, in_d]);
        if packed.numel() != quant.numel() { return Err(LmError::ShapeMismatch); }
        packed.dequant_row(0, 1.0f32, &mut []);
        quant.dequant_into(&mut []);
        matvec_w8(&inp[..in_d], q.q, q.scales, in_d, out_d, &mut out[..out_d])
            .map_err(|_| LmError::ShapeMismatch)?;
        return Ok(());
    }
    if w.len() < out_d * in_d {
        return Err(LmError::ShapeMismatch);
    }
    if crate::engine::try_invoke_gpu_kernel::<T>(
        inp,
        out,
        1,
        out_d,
        in_d,
        out_d,
        w,
        &[],
        crate::base::activations::ActivationKind::Identity,
    ) {
        return Ok(());
    }
    let out_ptr = crate::engine::runtime::SyncMutPtr(out.as_mut_ptr());
    let inp_s = inp;
    let w_s = w;
    crate::engine::runtime::parallel_for(out_d, in_d, &move |o| {
        let _ = &out_ptr;
        let row = &w_s[o * in_d..(o + 1) * in_d];
        unsafe { *out_ptr.0.add(o) = crate::engine::runtime::dot::<T>(inp_s, row, in_d); }
    });
    Ok(())
}

#[inline]
fn silu<T: Float>(x: T) -> T {
    x / (T::ONE + (-x).exp())
}

#[inline]
pub fn silu_prime<T: Float>(x: T) -> T {
    let s = T::ONE / (T::ONE + (-x).exp());
    s * (T::ONE + x * (T::ONE - s))
}

pub struct BlockQuant<'a, 'b, T> {
    pub qw: Option<&'a BlockQWeights<'b, T>>,
    pub tw: Option<&'a BlockTWeights<'b, T>>,
}

pub fn transformer_block_forward<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmBlockWeights<T>,
    position: usize,
    kv: &mut KvCacheView<T>,
    sc: &mut ForwardScratch<T>,
    quant: BlockQuant<'_, '_, T>,
    mut acts: Option<&mut BlockActivations<T>>,
    lora: Option<(&LmBlockLora<T>, &mut [T])>,
) -> Result<(), LmError> {
    let BlockQuant { qw, tw } = quant;
    let mut lora_opt = lora;
    let h = cfg.hidden_size;
    let head_dim = cfg.head_dim()?;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let kv_h = cfg.kv_h();
    let groups = cfg.kv_head_groups();
    let ffw = cfg.ffw_size;

    if sc.hidden.len() < h
        || sc.norm_buf.len() < h
        || sc.qkv.len() < h + 2 * kv_h
        || sc.wo_out.len() < h
        || sc.attn_head.len() < h
        || sc.ffn_gate.len() < ffw
        || sc.ffn_up.len() < ffw
        || sc.ffn_out.len() < h
    {
        return Err(LmError::ScratchTooSmall);
    }

    if let Some(ref mut acts) = acts {
        if acts.pre_norm_attn.len() >= h {
            acts.pre_norm_attn[..h].copy_from_slice(&sc.hidden[..h]);
        }
    }

    sc.norm_buf[..h].copy_from_slice(&sc.hidden[..h]);
    rms_norm_in_place(&mut sc.norm_buf[..h], w.attn_norm_gamma, T::from_f32(1e-5))
        .map_err(|_| LmError::ShapeMismatch)?;

    if let Some(ref mut acts) = acts {
        if acts.post_norm_attn.len() >= h {
            acts.post_norm_attn[..h].copy_from_slice(&sc.norm_buf[..h]);
        }
    }

    {
        let norm_x = &sc.norm_buf[..h];
        let (q_buf, rest) = sc.qkv.split_at_mut(h);
        let (k_buf, v_buf) = rest.split_at_mut(kv_h);
        matmul(norm_x, w.wq, qw.map(|q| q.wq), tw.map(|t| t.wq), h, h, q_buf)?;
        if let Some((ref lo, ref mut tmp)) = lora_opt {
            if let (Some(a), Some(b)) = (lo.wq_a, lo.wq_b) {
                lora_forward_delta(norm_x, a, b, q_buf, tmp, 1, h, h, lo.rank, lo.alpha)
                    .map_err(|_| LmError::ShapeMismatch)?;
            }
        }
        matmul(norm_x, w.wk, qw.map(|q| q.wk), tw.map(|t| t.wk), h, kv_h, k_buf)?;
        if let Some((ref lo, ref mut tmp)) = lora_opt {
            if let (Some(a), Some(b)) = (lo.wk_a, lo.wk_b) {
                lora_forward_delta(norm_x, a, b, k_buf, tmp, 1, h, kv_h, lo.rank, lo.alpha)
                    .map_err(|_| LmError::ShapeMismatch)?;
            }
        }
        matmul(norm_x, w.wv, qw.map(|q| q.wv), tw.map(|t| t.wv), h, kv_h, v_buf)?;
        if let Some((ref lo, ref mut tmp)) = lora_opt {
            if let (Some(a), Some(b)) = (lo.wv_a, lo.wv_b) {
                lora_forward_delta(norm_x, a, b, v_buf, tmp, 1, h, kv_h, lo.rank, lo.alpha)
                    .map_err(|_| LmError::ShapeMismatch)?;
            }
        }
    }

    for head in 0..nh {
        let off = head * head_dim;
        apply_rope_in_place(&mut sc.qkv[off..off + head_dim], position, cfg.rope_theta)
            .map_err(|_| LmError::ShapeMismatch)?;
    }
    for kv_head in 0..nkv {
        let k_off = h + kv_head * head_dim;
        apply_rope_in_place(&mut sc.qkv[k_off..k_off + head_dim], position, cfg.rope_theta)
            .map_err(|_| LmError::ShapeMismatch)?;
    }

    if let Some(ref mut acts) = acts {
        if acts.q.len() >= h {
            acts.q[..h].copy_from_slice(&sc.qkv[..h]);
        }
    }

    kv.append_token(&sc.qkv[h..h + kv_h], &sc.qkv[h + kv_h..h + 2 * kv_h])
        .map_err(|_| LmError::KvCapacityExceeded)?;

    let seq_len = kv.len_tokens();
    let kv_stride = kv_h;

    if let Some(ref mut acts) = acts {
        if let Ok((k_cached, v_cached)) = kv.token_slices(seq_len - 1) {
            if acts.k_proj.len() >= kv_h {
                acts.k_proj[..kv_h].copy_from_slice(&k_cached[..kv_h]);
            }
            if acts.v_proj.len() >= kv_h {
                acts.v_proj[..kv_h].copy_from_slice(&v_cached[..kv_h]);
            }
        }
    }

    if sc.scores.len() < seq_len {
        return Err(LmError::ScratchTooSmall);
    }

    sc.attn_head[..h].iter_mut().for_each(|v| *v = T::ZERO);

    for head in 0..nh {
        let q_off = head * head_dim;
        let kv_head = head / groups;
        let kv_head_off = kv_head * head_dim;
        let scores_slice = &mut sc.scores[..seq_len];

        sdpa_strided(
            &sc.qkv[q_off..q_off + head_dim],
            &kv.key[..seq_len * kv_stride],
            &kv.value[..seq_len * kv_stride],
            StridedHead { head_offset: kv_head_off, head_dim, seq_len, stride: kv_stride },
            &mut sc.attn_head[q_off..q_off + head_dim],
            scores_slice,
        )
        .map_err(|_| LmError::ShapeMismatch)?;

        if acts.is_some() && head == 0 && seq_len > 1 {
            let sparse = crate::base::tensor::SparseTensor::<T>::from_raw(
                core::ptr::null(),
                core::ptr::null(),
                core::ptr::null(),
                0,
                seq_len,
                seq_len,
            );
            if !sparse.row_indices.is_null() || sparse.shape[0] != seq_len {
                let tmp_x = [T::ZERO; 1];
                let mut tmp_y = [T::ZERO; 1];
                sparse.spmv(&tmp_x, &mut tmp_y);
                sparse.apply_mask(&mut tmp_y);
            }
            let mut probe_out = [T::ZERO; 1];
            let _ = sdpa_with_sparse_mask(
                &sc.qkv[q_off..q_off + head_dim],
                &kv.key[..seq_len * kv_stride],
                &kv.value[..seq_len * kv_stride],
                crate::graph::attn::attention::AttentionShape {
                    q_len: 1, k_len: seq_len, d_k: head_dim, d_v: head_dim,
                    k_stride: kv_stride, v_stride: kv_stride, head_offset: kv_head_off,
                },
                &mut probe_out,
                scores_slice,
                &sparse,
            );
        }

        if let Some(ref mut acts) = acts {
            let dst_off = head * cfg.context_len;
            if acts.attn_scores_per_head.len() >= dst_off + seq_len {
                acts.attn_scores_per_head[dst_off..dst_off + seq_len]
                    .copy_from_slice(&sc.scores[..seq_len]);
            }
        }
    }

    if let Some(ref mut acts) = acts {
        if acts.pre_wo.len() >= h {
            acts.pre_wo[..h].copy_from_slice(&sc.attn_head[..h]);
        }
        acts.seq_len = seq_len;
    }

    matmul(&sc.attn_head[..h], w.wo, qw.map(|q| q.wo), tw.map(|t| t.wo), h, h, &mut sc.wo_out[..h])?;
    if let Some((ref lo, ref mut tmp)) = lora_opt {
        if let (Some(a), Some(b)) = (lo.wo_a, lo.wo_b) {
            lora_forward_delta(&sc.attn_head[..h], a, b, &mut sc.wo_out[..h], tmp, 1, h, h, lo.rank, lo.alpha)
                .map_err(|_| LmError::ShapeMismatch)?;
        }
    }
    for i in 0..h {
        sc.hidden[i] += sc.wo_out[i];
    }

    if let Some(ref mut acts) = acts {
        if acts.pre_norm_ffn.len() >= h {
            acts.pre_norm_ffn[..h].copy_from_slice(&sc.hidden[..h]);
        }
    }

    sc.norm_buf[..h].copy_from_slice(&sc.hidden[..h]);
    rms_norm_in_place(&mut sc.norm_buf[..h], w.ffn_norm_gamma, T::from_f32(1e-5))
        .map_err(|_| LmError::ShapeMismatch)?;

    if let Some(ref mut acts) = acts {
        if acts.post_norm_ffn.len() >= h {
            acts.post_norm_ffn[..h].copy_from_slice(&sc.norm_buf[..h]);
        }
    }

    if let Some(moe) = w.moe {
        let ne = moe.num_experts;
        let per = 3 * ffw * h;
        if ne == 0 || moe.router.len() < ne * h || moe.experts.len() < ne * per {
            return Err(LmError::ShapeMismatch);
        }
        let score_bytes = 2 * ne * core::mem::size_of::<f32>();
        let base = crate::engine::runtime::hardware::mmap_shared_anon(score_bytes);
        if base.is_null() {
            return Err(LmError::ScratchTooSmall);
        }
        let scores = unsafe { core::slice::from_raw_parts_mut(base.cast::<f32>(), ne) };
        let expert_frac = unsafe { core::slice::from_raw_parts_mut(base.cast::<f32>().add(ne), ne) };

        let mut run = || -> Result<(), LmError> {
            let norm_x2 = &sc.norm_buf[..h];
            for (e, score) in scores.iter_mut().enumerate().take(ne) {
                let rw = &moe.router[e * h..e * h + h];
                let mut acc = 0.0f64;
                for i in 0..h {
                    acc += norm_x2[i].to_f64() * rw[i].to_f64();
                }
                *score = acc as f32;
            }
            crate::graph::blocks::moe::softmax_scores_inplace(scores, ne).map_err(|_| LmError::ShapeMismatch)?;
            for e in 0..ne { expert_frac[e] = 0.0; }

            if ne >= 2 {
                let (e1, e2, w1, w2) = crate::graph::blocks::moe::top2_gating_weighted(scores, ne)
                    .map_err(|_| LmError::ShapeMismatch)?;
                if e1 < ne { expert_frac[e1] = 0.5; }
                if e2 < ne { expert_frac[e2] = 0.5; }

                let be1 = e1 * per;
                matmul(norm_x2, &moe.experts[be1..be1 + ffw * h], None, None, h, ffw, &mut sc.ffn_gate[..ffw])?;
                matmul(norm_x2, &moe.experts[be1 + ffw * h..be1 + 2 * ffw * h], None, None, h, ffw, &mut sc.ffn_up[..ffw])?;
                for i in 0..ffw { sc.ffn_gate[i] = silu(sc.ffn_gate[i]) * sc.ffn_up[i]; }
                matmul(&sc.ffn_gate[..ffw], &moe.experts[be1 + 2 * ffw * h..be1 + per], None, None, ffw, h, &mut sc.wo_out[..h])?;

                let be2 = e2 * per;
                matmul(norm_x2, &moe.experts[be2..be2 + ffw * h], None, None, h, ffw, &mut sc.ffn_gate[..ffw])?;
                matmul(norm_x2, &moe.experts[be2 + ffw * h..be2 + 2 * ffw * h], None, None, h, ffw, &mut sc.ffn_up[..ffw])?;
                for i in 0..ffw { sc.ffn_gate[i] = silu(sc.ffn_gate[i]) * sc.ffn_up[i]; }
                matmul(&sc.ffn_gate[..ffw], &moe.experts[be2 + 2 * ffw * h..be2 + per], None, None, ffw, h, &mut sc.attn_head[..h])?;

                crate::graph::blocks::moe::route_top2_weighted(
                    &sc.wo_out[..h], &sc.attn_head[..h], h, w1, w2, &mut sc.ffn_out[..h],
                ).map_err(|_| LmError::ShapeMismatch)?;
            } else {
                let chosen = crate::graph::blocks::moe::top1_gating(scores, ne).map_err(|_| LmError::ShapeMismatch)?;
                if chosen < ne { expert_frac[chosen] = 1.0; }
                let be = chosen * per;
                matmul(norm_x2, &moe.experts[be..be + ffw * h], None, None, h, ffw, &mut sc.ffn_gate[..ffw])?;
                matmul(norm_x2, &moe.experts[be + ffw * h..be + 2 * ffw * h], None, None, h, ffw, &mut sc.ffn_up[..ffw])?;
                for i in 0..ffw { sc.ffn_gate[i] = silu(sc.ffn_gate[i]) * sc.ffn_up[i]; }
                matmul(&sc.ffn_gate[..ffw], &moe.experts[be + 2 * ffw * h..be + per], None, None, ffw, h, &mut sc.wo_out[..h])?;
                crate::graph::blocks::moe::route_top1(&sc.wo_out[..h], h, &mut sc.ffn_out[..h])
                    .map_err(|_| LmError::ShapeMismatch)?;
            }

            sc.moe_aux_loss += crate::graph::blocks::moe::moe_load_aux_loss(expert_frac, scores, ne);

            if acts.is_some() && ne <= 16 {
                let ra_scratch_len = ne * h + 3 * ne * ne + 2 * ne;
                if let Some(mut dense) = crate::base::tensor::DenseTensor::<T>::alloc(
                    [1, 1, 1, 1, ra_scratch_len]
                ) {
                    if dense.is_empty() || dense.len() < ra_scratch_len || dense.shape[4] != ra_scratch_len {
                        return Err(LmError::ScratchTooSmall);
                    }
                    if dense.as_slice().len() != ra_scratch_len {
                        return Err(LmError::ScratchTooSmall);
                    }
                    let ra_buf = dense.as_mut_slice();
                    let mut evals_t = crate::base::tensor::StaticTensor::<T, 16>::zeroed([1, 1, 1, 1, 16]);
                    let mut evecs_t = crate::base::tensor::StaticTensor::<T, 256>::zeroed([1, 1, 1, 1, 256]);
                    let ne_capped = ne.min(evals_t.shape[4]);
                    evals_t.used = ne_capped;
                    evals_t.fill_zero();
                    let evals_len = evals_t.as_slice().len();
                    let evecs_len = evecs_t.as_mut_slice().len();
                    if evals_len < ne_capped || evecs_len < ne_capped * ne_capped {
                        return Err(LmError::ScratchTooSmall);
                    }
                    let cond = crate::graph::blocks::moe::moe_router_condition(moe.router, ne, h, ra_buf);
                    let _pd = crate::graph::blocks::moe::moe_router_is_pd(moe.router, ne, h, ra_buf);
                    crate::graph::blocks::moe::moe_router_eigenvectors(moe.router, ne, h, &mut evals_t.data[..ne_capped], &mut evecs_t.data[..ne_capped*ne_capped], ra_buf);
                    let _det = crate::graph::blocks::moe::moe_router_determinant(moe.router, ne, h, ra_buf);
                    if cond > T::ZERO {
                        let mut inv_out = crate::base::tensor::StaticTensor::<T, 256>::zeroed([1, 1, 1, ne, ne]);
                        crate::graph::blocks::moe::moe_router_inverse_gram(moe.router, ne, h, &mut inv_out.data[..ne*ne], ra_buf);
                    }
                }
            }

            for i in 0..h {
                sc.hidden[i] += sc.ffn_out[i];
            }
            Ok(())
        };

        let r = run();
        crate::engine::runtime::hardware::munmap(base, score_bytes);
        r?;
        return Ok(());
    }

    {
        let norm_x2 = &sc.norm_buf[..h];
        matmul(norm_x2, w.w_gate, qw.map(|q| q.w_gate), tw.map(|t| t.w_gate), h, ffw, &mut sc.ffn_gate[..ffw])?;
        matmul(norm_x2, w.w_up, qw.map(|q| q.w_up), tw.map(|t| t.w_up), h, ffw, &mut sc.ffn_up[..ffw])?;
    }

    if let Some(ref mut acts) = acts {
        if acts.gate_pre.len() >= ffw {
            acts.gate_pre[..ffw].copy_from_slice(&sc.ffn_gate[..ffw]);
        }
        if acts.up.len() >= ffw {
            acts.up[..ffw].copy_from_slice(&sc.ffn_up[..ffw]);
        }
    }

    for i in 0..ffw {
        sc.ffn_gate[i] = silu(sc.ffn_gate[i]) * sc.ffn_up[i];
    }

    matmul(&sc.ffn_gate[..ffw], w.w_down, qw.map(|q| q.w_down), tw.map(|t| t.w_down), ffw, h, &mut sc.ffn_out[..h])?;

    for i in 0..h {
        sc.hidden[i] += sc.ffn_out[i];
    }

    Ok(())
}
