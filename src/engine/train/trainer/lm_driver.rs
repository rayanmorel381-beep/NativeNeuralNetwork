use crate::engine::rnn_flow::MAX_LAYERS;
use crate::engine::infer::inference::{
    batched_work_count, block_param_count, checkpoint_count, forward_loss_batched, train_batched,
    BackwardScratch, BlockActivations, ForwardScratch, TrainBatchedBuffers,
};
use crate::graph::attn::kv_cache::KvCacheView;
use crate::graph::lm::{
    accumulate_grads, accumulate_grads_masked, apply_grads_adamw, lm_grads_carve_into,
    lm_weights_param_count, lm_weights_read_into, train_activations_buffer_count,
    train_activations_carve_into, AdamState, Lcg, LmBlockGrads,
    LmBlockWeights, TrainScratch,
};
use crate::engine::train::trainer::lm_step::{train_step, train_step_masked, LmTrainMode};
use crate::api::rnn_api::core_api::RnnApiError;
use core::mem::MaybeUninit;

pub struct LmTrainBufs<'a, T> {
    pub params: &'a mut [T],
    pub grads_buf: &'a mut [T],
    pub m_buf: &'a mut [T],
    pub v_buf: &'a mut [T],
    pub work_buf: &'a mut [T],
    pub checkpoints_buf: &'a mut [T],
    pub fwd_buf: &'a mut [T],
    pub acts_buf: &'a mut [T],
    pub back_tmp1: &'a mut [T],
    pub back_tmp2: &'a mut [T],
    pub back_dscores: &'a mut [T],
    pub kv_buf: &'a mut [T],
    pub metrics_out: &'a mut [T],
}

pub struct LmTrainRun<'a> {
    pub target_active: &'a [bool],
    pub seq_lens: &'a [usize],
    pub mode: LmTrainMode,
    pub train_window: usize,
    pub num_steps: usize,
    pub accum_micro: usize,
    pub seed: u64,
    pub distill_targets: Option<&'a [u32]>,
    pub deadline_ns: u64,
}

fn train_lm_batched<T: crate::base::math::Float + 'static>(
    cfg: &crate::graph::lm::LmConfig<T>,
    token_ids: &[u32],
    train_cfg: &crate::graph::lm::LmTrainConfig<T>,
    bufs: &mut LmTrainBufs<'_, T>,
    run: &LmTrainRun<'_>,
) -> Result<(f64, usize), RnnApiError> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let window = run.train_window;
    let param_count = lm_weights_param_count(cfg);
    if bufs.params.len() < param_count
        || bufs.m_buf.len() < param_count
        || bufs.v_buf.len() < param_count
    {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let bpc = block_param_count(cfg);
    if bufs.grads_buf.len() < bpc + v * h + h {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if bufs.work_buf.len() < batched_work_count(cfg) {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if bufs.checkpoints_buf.len() < checkpoint_count(cfg) {
        return Err(RnnApiError::CapacityTooSmall);
    }

    let local_train_cfg = *train_cfg;
    let max_start = token_ids.len() - window - 1;
    let mut window_rng = Lcg::new(run.seed);
    let mut total_loss = 0.0f64;
    let mut done = 0usize;
    let mut last_grad_norm = T::ZERO;
    let mut last_lr = T::ZERO;
    let mut total_ops = crate::observability::profiler::OpCounter::new();
    let train_start_ns = crate::engine::runtime::hardware::monotonic_ns();

    for s in 0..run.num_steps {
        if run.deadline_ns != 0 && crate::engine::runtime::hardware::monotonic_ns() >= run.deadline_ns {
            break;
        }
        let step = (s as u32).wrapping_add(1);
        let start = (window_rng.next_u64() as usize) % (max_start + 1);
        let token_window = &token_ids[start..start + window + 1];

        let distill_window = run
            .distill_targets
            .map(|dt| &dt[start..start + window]);

        let (grad_layer, rest) = bufs.grads_buf.split_at_mut(bpc);
        let (embed_grad, rest2) = rest.split_at_mut(v * h);
        let (final_grad, _) = rest2.split_at_mut(h);

        let mut tb = TrainBatchedBuffers {
            params: &mut *bufs.params,
            m: &mut *bufs.m_buf,
            v: &mut *bufs.v_buf,
            grad_layer,
            embed_grad,
            final_grad,
            work: &mut *bufs.work_buf,
            checkpoints: &mut *bufs.checkpoints_buf,
        };

        let outcome = train_batched(cfg, &local_train_cfg, &mut tb, token_window, None, distill_window, step)
            .map_err(|_| RnnApiError::Model)?;
        total_loss += outcome.loss.to_f64();
        last_grad_norm = outcome.grad_norm;
        last_lr = outcome.lr;
        total_ops.merge(&outcome.ops);
        done += 1;
    }

    if bufs.metrics_out.len() >= 2 {
        bufs.metrics_out[0] = last_grad_norm;
        bufs.metrics_out[1] = last_lr;
    }
    if bufs.metrics_out.len() >= 3 {
        let train_secs = crate::engine::runtime::hardware::monotonic_ns()
            .saturating_sub(train_start_ns) as f32
            / 1_000_000_000.0;
        bufs.metrics_out[2] = T::from_f32(crate::observability::profiler::ops_per_second(&total_ops, train_secs));
    }

    if done == 0 {
        return Ok((0.0, 0));
    }
    let avg = total_loss / done as f64;
    let val_window = &token_ids[0..window + 1];
    match forward_loss_batched(cfg, &*bufs.params, val_window, bufs.work_buf) {
        Ok(loss) => Ok((loss.to_f64(), done)),
        Err(_) => Ok((avg, done)),
    }
}

fn train_lm_sequential<T: crate::base::math::Float + 'static>(
    cfg: &crate::graph::lm::LmConfig<T>,
    token_ids: &[u32],
    train_cfg: &crate::graph::lm::LmTrainConfig<T>,
    bufs: &mut LmTrainBufs<'_, T>,
    run: &LmTrainRun<'_>,
    masked: bool,
    accumulate: bool,
) -> Result<(f64, usize), RnnApiError> {
    let nl = cfg.num_layers;
    if nl == 0 || nl > MAX_LAYERS {
        return Err(RnnApiError::BadBytes);
    }

    let params = &mut *bufs.params;
    let grads_buf = &mut *bufs.grads_buf;
    let m_buf = &mut *bufs.m_buf;
    let v_buf = &mut *bufs.v_buf;
    let work_buf = &mut *bufs.work_buf;
    let fwd_buf = &mut *bufs.fwd_buf;
    let acts_buf = &mut *bufs.acts_buf;
    let back_tmp1 = &mut *bufs.back_tmp1;
    let back_tmp2 = &mut *bufs.back_tmp2;
    let back_dscores = &mut *bufs.back_dscores;
    let kv_buf = &mut *bufs.kv_buf;
    let window = run.train_window;
    let target_active = run.target_active;

    let h = cfg.hidden_size;
    let ffw = cfg.ffw_size;
    let kv_h = cfg.kv_h();
    let ctx = cfg.context_len;
    let head_dim = cfg.head_dim().map_err(|_| RnnApiError::BadBytes)?;

    let param_count = lm_weights_param_count(cfg);
    if params.len() < param_count
        || grads_buf.len() < param_count
        || m_buf.len() < param_count
        || v_buf.len() < param_count
    {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if fwd_buf.len() < ForwardScratch::total_f32_count(cfg) {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if acts_buf.len() < train_activations_buffer_count(cfg) {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let back_tmp_need = if h > ffw { h } else { ffw };
    if back_tmp1.len() < back_tmp_need || back_tmp2.len() < back_tmp_need {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if back_dscores.len() < window + 2 * kv_h {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let kv_stride = ctx * kv_h;
    if kv_buf.len() < nl * 2 * kv_stride {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if masked && target_active.len() + 1 < token_ids.len() {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let micro = if accumulate {
        if work_buf.len() < param_count {
            return Err(RnnApiError::CapacityTooSmall);
        }
        if run.accum_micro == 0 {
            return Err(RnnApiError::BadBytes);
        }
        run.accum_micro
    } else {
        1
    };

    let params_ptr = params.as_mut_ptr();
    let params_len = params.len();
    let grads_ptr = grads_buf.as_mut_ptr();
    let grads_len = grads_buf.len();
    let m_ptr = m_buf.as_mut_ptr();
    let m_len = m_buf.len();
    let v_ptr = v_buf.as_mut_ptr();
    let v_len = v_buf.len();
    let work_ptr = work_buf.as_mut_ptr();
    let fwd_ptr = fwd_buf.as_mut_ptr();
    let fwd_len = fwd_buf.len();
    let acts_ptr = acts_buf.as_mut_ptr();
    let acts_len = acts_buf.len();
    let bt1_ptr = back_tmp1.as_mut_ptr();
    let bt1_len = back_tmp1.len();
    let bt2_ptr = back_tmp2.as_mut_ptr();
    let bt2_len = back_tmp2.len();
    let bds_ptr = back_dscores.as_mut_ptr();
    let bds_len = back_dscores.len();
    let kv_ptr = kv_buf.as_mut_ptr();
    let kv_len = kv_buf.len();
    let tokens_ptr = token_ids.as_ptr();
    let tokens_len = token_ids.len();

    let local_train_cfg = *train_cfg;
    let max_start = tokens_len - window - 1;
    let mut window_rng = Lcg::new(run.seed);

    let mut total_loss = 0.0f64;
    let mut done = 0usize;
    let mut last_grad_norm = T::ZERO;
    let mut last_lr = T::ZERO;

    for s in 0..run.num_steps {
        let step = (s as u32).wrapping_add(1);

        if accumulate {
            let accum: &mut [T] = unsafe { core::slice::from_raw_parts_mut(work_ptr, param_count) };
            for a in accum.iter_mut() {
                *a = T::ZERO;
            }
        }

        let mut step_loss = 0.0f64;
        let mut micro_done = 0usize;

        for _ in 0..micro {
            let params_ro: &[T] = unsafe { core::slice::from_raw_parts(params_ptr, params_len) };
            let params_mut: &mut [T] = unsafe { core::slice::from_raw_parts_mut(params_ptr, params_len) };
            let grads_for_carve: &mut [T] = unsafe { core::slice::from_raw_parts_mut(grads_ptr, grads_len) };
            let grads_for_adam: &mut [T] = unsafe { core::slice::from_raw_parts_mut(grads_ptr, grads_len) };
            let m_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(m_ptr, m_len) };
            let v_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(v_ptr, v_len) };
            let fwd_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(fwd_ptr, fwd_len) };
            let acts_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(acts_ptr, acts_len) };
            let bt1_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(bt1_ptr, bt1_len) };
            let bt2_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(bt2_ptr, bt2_len) };
            let bds_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(bds_ptr, bds_len) };
            let kv_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(kv_ptr, kv_len) };

            let mut blocks_w: [MaybeUninit<LmBlockWeights<'_, T>>; MAX_LAYERS] =
                [const { MaybeUninit::uninit() }; MAX_LAYERS];
            let mut blocks_g: [MaybeUninit<LmBlockGrads<'_, T>>; MAX_LAYERS] =
                [const { MaybeUninit::uninit() }; MAX_LAYERS];
            let mut blocks_acts: [MaybeUninit<BlockActivations<'_, T>>; MAX_LAYERS] =
                [const { MaybeUninit::uninit() }; MAX_LAYERS];
            for i in 0..nl {
                blocks_w[i].write(LmBlockWeights {
                    attn_norm_gamma: &[], wq: &[], wk: &[], wv: &[], wo: &[],
                    ffn_norm_gamma: &[], w_gate: &[], w_up: &[], w_down: &[],
                    moe: None,
                });
                blocks_g[i].write(LmBlockGrads {
                    attn_norm_gamma: &mut [], wq: &mut [], wk: &mut [], wv: &mut [], wo: &mut [],
                    ffn_norm_gamma: &mut [], w_gate: &mut [], w_up: &mut [], w_down: &mut [],
                });
                blocks_acts[i].write(BlockActivations {
                    pre_norm_attn: &mut [], post_norm_attn: &mut [], q: &mut [],
                    k_proj: &mut [], v_proj: &mut [], attn_scores_per_head: &mut [],
                    pre_wo: &mut [], pre_norm_ffn: &mut [], post_norm_ffn: &mut [],
                    gate_pre: &mut [], up: &mut [], seq_len: 0,
                });
            }
            let blocks_w: &mut [LmBlockWeights<'_, T>] =
                unsafe { core::slice::from_raw_parts_mut(blocks_w[0].as_mut_ptr(), nl) };
            let blocks_g: &mut [LmBlockGrads<'_, T>] =
                unsafe { core::slice::from_raw_parts_mut(blocks_g[0].as_mut_ptr(), nl) };
            let blocks_acts: &mut [BlockActivations<'_, T>] =
                unsafe { core::slice::from_raw_parts_mut(blocks_acts[0].as_mut_ptr(), nl) };

            let weights = lm_weights_read_into(cfg, params_ro, blocks_w).map_err(|_| RnnApiError::BadBytes)?;
            let mut grads = lm_grads_carve_into(grads_for_carve, cfg, blocks_g).map_err(|_| RnnApiError::BadBytes)?;
            let mut train_acts =
                train_activations_carve_into(acts_local, cfg, blocks_acts).map_err(|_| RnnApiError::BadBytes)?;

            let (hidden, rest) = fwd_local.split_at_mut(h);
            let (norm_buf, rest) = rest.split_at_mut(h);
            let (qkv, rest) = rest.split_at_mut(h + 2 * kv_h);
            let (wo_out, rest) = rest.split_at_mut(h);
            let (scores, rest) = rest.split_at_mut(ctx);
            let (attn_head, rest) = rest.split_at_mut(h);
            let (ffn_gate, rest) = rest.split_at_mut(ffw);
            let (ffn_up, rest) = rest.split_at_mut(ffw);
            let (ffn_out, rest) = rest.split_at_mut(h);
            let (logits, rest) = rest.split_at_mut(cfg.vocab_size);
            let hh = cfg.head_hidden;
            let (head_z1, rest) = rest.split_at_mut(hh);
            let (head_a1, _) = rest.split_at_mut(hh);
            let mut sc = ForwardScratch {
                hidden, norm_buf, qkv, wo_out, scores, attn_head, ffn_gate, ffn_up, ffn_out, logits, head_z1, head_a1, moe_aux_loss: 0.0,
            };
            let mut back_sc = BackwardScratch { tmp1: bt1_local, tmp2: bt2_local, d_scores: bds_local };

            let mut kv_uninit: [MaybeUninit<KvCacheView<'_, T>>; MAX_LAYERS] =
                [const { MaybeUninit::uninit() }; MAX_LAYERS];
            let mut kv_remaining = &mut kv_local[..];
            for slot in kv_uninit.iter_mut().take(nl) {
                let (key_part, r) = kv_remaining.split_at_mut(kv_stride);
                let (val_part, next) = r.split_at_mut(kv_stride);
                kv_remaining = next;
                slot.write(KvCacheView {
                    key: key_part,
                    value: val_part,
                    max_tokens: ctx,
                    head_dim,
                    num_heads: cfg.num_kv_heads,
                    used_tokens: 0,
                });
            }
            let kv_layers: &mut [KvCacheView<'_, T>] =
                unsafe { core::slice::from_raw_parts_mut(kv_uninit[0].as_mut_ptr(), nl) };

            let start = (window_rng.next_u64() as usize) % (max_start + 1);
            let token_window: &[u32] = unsafe { core::slice::from_raw_parts(tokens_ptr.add(start), window + 1) };

            let adam = AdamState {
                params: params_mut,
                grads_buf: grads_for_adam,
                m: m_local,
                v: v_local,
                step,
            };
            let scratch = TrainScratch {
                kv_layers,
                sc: &mut sc,
                train_acts: &mut train_acts,
                back_sc: &mut back_sc,
            };

            let outcome = if accumulate {
                if masked {
                    let mask_window = &target_active[start..start + window];
                    accumulate_grads_masked(
                        cfg, &local_train_cfg, adam, &weights, &mut grads, token_window,
                        mask_window, scratch,
                    )
                } else {
                    accumulate_grads(cfg, &local_train_cfg, adam, &weights, &mut grads, token_window, scratch)
                }
            } else if masked {
                let mask_window = &target_active[start..start + window];
                train_step_masked(
                    cfg, &local_train_cfg, adam, &weights, &mut grads, token_window, mask_window,
                    scratch,
                )
            } else {
                train_step(cfg, &local_train_cfg, adam, &weights, &mut grads, token_window, scratch)
            }
            .map_err(|_| RnnApiError::Model)?;

            step_loss += outcome.loss.to_f64();
            last_grad_norm = outcome.grad_norm;
            last_lr = outcome.lr;
            micro_done += 1;

            if accumulate {
                let grads_src: &[T] = unsafe { core::slice::from_raw_parts(grads_ptr, param_count) };
                let accum: &mut [T] = unsafe { core::slice::from_raw_parts_mut(work_ptr, param_count) };
                for (a, &g) in accum.iter_mut().zip(grads_src.iter()) {
                    *a += g;
                }
            }
        }

        if accumulate {
            {
                let accum: &mut [T] = unsafe { core::slice::from_raw_parts_mut(work_ptr, param_count) };
                let inv = T::ONE / T::from_usize(micro);
                for a in accum.iter_mut() {
                    *a *= inv;
                }
            }
            let params_mut: &mut [T] = unsafe { core::slice::from_raw_parts_mut(params_ptr, param_count) };
            let grads_mut: &mut [T] = unsafe { core::slice::from_raw_parts_mut(work_ptr, param_count) };
            let m_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(m_ptr, param_count) };
            let v_local: &mut [T] = unsafe { core::slice::from_raw_parts_mut(v_ptr, param_count) };
            apply_grads_adamw(&local_train_cfg, params_mut, grads_mut, m_local, v_local, step)
                .map_err(|_| RnnApiError::Model)?;
        }

        total_loss += if micro_done == 0 { 0.0 } else { step_loss / micro_done as f64 };
        done += 1;
    }

    let avg = if done == 0 { 0.0 } else { total_loss / done as f64 };
    if bufs.metrics_out.len() >= 2 {
        bufs.metrics_out[0] = last_grad_norm;
        bufs.metrics_out[1] = last_lr;
    }
    Ok((avg, done))
}

fn train_lm_multiseq<T: crate::base::math::Float + 'static>(
    cfg: &crate::graph::lm::LmConfig<T>,
    token_ids: &[u32],
    train_cfg: &crate::graph::lm::LmTrainConfig<T>,
    bufs: &mut LmTrainBufs<'_, T>,
    run: &LmTrainRun<'_>,
) -> Result<(f64, usize), RnnApiError> {
    let window = run.train_window;
    let nseq = run.seq_lens.len();
    if nseq == 0 {
        return Err(RnnApiError::BadBytes);
    }
    let mut bounds: [usize; 64] = [0; 64];
    if nseq > bounds.len() {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let mut acc = 0usize;
    for (i, &length) in run.seq_lens.iter().take(nseq).enumerate() {
        acc = acc.checked_add(length).ok_or(RnnApiError::BadBytes)?;
        bounds[i] = acc;
    }
    if acc > token_ids.len() {
        return Err(RnnApiError::BadBytes);
    }
    let mut seqs: [&[u32]; 64] = [&[]; 64];
    let mut start = 0usize;
    for (i, &bound) in bounds.iter().take(nseq).enumerate() {
        seqs[i] = &token_ids[start..bound];
        start = bound;
    }
    let mut lens: [usize; 64] = [0; 64];
    if !crate::engine::train::batching::sequence_lengths(&seqs[..nseq], &mut lens[..nseq]) {
        return Err(RnnApiError::BadBytes);
    }
    for (i, &expected_len) in run.seq_lens.iter().take(nseq).enumerate() {
        if lens[i] != expected_len {
            return Err(RnnApiError::BadBytes);
        }
    }
    let pad_id = train_cfg.pad_id;
    let max_len = crate::engine::train::batching::max_sequence_len(&seqs[..nseq]).ok_or(RnnApiError::BadBytes)?;
    let row = window + 1;
    if max_len > row {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let pad_count = nseq * row;
    let pad_bytes = pad_count * core::mem::size_of::<u32>();
    let mask_count = nseq * window;
    if core::mem::size_of_val(bufs.work_buf) < pad_bytes + pad_count + mask_count {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let work_ptr = bufs.work_buf.as_mut_ptr() as *mut u8;
    let pad_ids: &mut [u32] = unsafe { core::slice::from_raw_parts_mut(work_ptr as *mut u32, pad_count) };
    match crate::engine::train::batching::pad_sequences_u32(&seqs[..nseq], pad_id, pad_ids, row) {
        Ok(()) => {}
        Err(crate::engine::train::batching::BatchError::Empty) => return Ok((0.0, 0)),
        Err(crate::engine::train::batching::BatchError::ShapeMismatch) => return Err(RnnApiError::CapacityTooSmall),
    }
    let full_mask: &mut [u8] = unsafe { core::slice::from_raw_parts_mut(work_ptr.add(pad_bytes), pad_count) };
    if crate::engine::train::batching::make_padding_mask(pad_ids, pad_id, full_mask).is_err() {
        return Err(RnnApiError::BadBytes);
    }
    let mask: &mut [bool] = unsafe { core::slice::from_raw_parts_mut(work_ptr.add(pad_bytes + pad_count) as *mut bool, mask_count) };
    let mut active_total = 0usize;
    crate::engine::train::batching::for_each_token_row(pad_ids, row, |r, tokens| {
        active_total += crate::engine::train::batching::count_non_pad(&tokens[1..], pad_id);
        for t in 0..window {
            mask[r * window + t] = full_mask[r * row + t + 1] != 0;
        }
    });
    if active_total == 0 {
        return Ok((0.0, 0));
    }
    let mut total_loss = 0.0f64;
    let mut done = 0usize;
    for s in 0..run.num_steps {
        let r = s % nseq;
        let pad_ids_ro: &[u32] = unsafe { core::slice::from_raw_parts(work_ptr as *const u32, pad_count) };
        let mask_ro: &[bool] = unsafe { core::slice::from_raw_parts(work_ptr.add(pad_bytes + pad_count) as *const bool, mask_count) };
        let window_ids = &pad_ids_ro[r * row..r * row + row];
        let active = &mask_ro[r * window..r * window + window];
        let seq_run = LmTrainRun {
            target_active: active,
            seq_lens: &[],
            mode: LmTrainMode::SequentialMasked,
            train_window: window,
            num_steps: 1,
            accum_micro: 1,
            seed: run.seed.wrapping_add(s as u64),
            distill_targets: None,
            deadline_ns: 0,
        };
        let (l, c) = train_lm_sequential(cfg, window_ids, train_cfg, bufs, &seq_run, true, false)?;
        total_loss += l;
        done += c;
    }
    if done == 0 {
        return Ok((0.0, 0));
    }
    Ok((total_loss / done as f64, done))
}

pub fn run_lm_train<T: crate::base::math::Float + 'static>(
    cfg: &crate::graph::lm::LmConfig<T>,
    token_ids: &[u32],
    train_cfg: &crate::graph::lm::LmTrainConfig<T>,
    bufs: &mut LmTrainBufs<'_, T>,
    run: &LmTrainRun<'_>,
) -> Result<(f64, usize), RnnApiError> {
    cfg.validate().map_err(|_| RnnApiError::BadBytes)?;
    if run.train_window < 1 || token_ids.len() < run.train_window + 1 {
        return Err(RnnApiError::BadBytes);
    }

    match run.mode {
        LmTrainMode::Batched => train_lm_batched(cfg, token_ids, train_cfg, bufs, run),
        LmTrainMode::Sequential => {
            train_lm_sequential(cfg, token_ids, train_cfg, bufs, run, false, false)
        }
        LmTrainMode::SequentialMasked => {
            train_lm_sequential(cfg, token_ids, train_cfg, bufs, run, true, false)
        }
        LmTrainMode::Accumulate => {
            train_lm_sequential(cfg, token_ids, train_cfg, bufs, run, false, true)
        }
        LmTrainMode::AccumulateMasked => {
            train_lm_sequential(cfg, token_ids, train_cfg, bufs, run, true, true)
        }
        LmTrainMode::MultiSequence => train_lm_multiseq(cfg, token_ids, train_cfg, bufs, run),
    }
}
