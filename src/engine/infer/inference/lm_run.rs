use crate::engine::rnn_flow::MAX_LAYERS;
use crate::api::rnn_api::core_api::RnnApiError;

pub struct LoraSpec<'a, T> {
    pub pack: &'a [T],
    pub rank: usize,
    pub alpha: T,
}

pub struct LmRunInputs<'a, T> {
    pub weights_buf: &'a [T],
    pub prompt_ids: &'a [u32],
    pub max_new_tokens: usize,
    pub sampling: &'a crate::graph::lm::SamplingConfig<T>,
    pub prompt_already_in_cache: bool,
    pub history: &'a [u32],
    pub stop_ids: &'a [u32],
    pub quant: crate::graph::lm::LmQuant,
    pub lora: Option<LoraSpec<'a, T>>,
}

pub struct LmRunBufs<'a, 'b, T> {
    pub out_ids: &'b mut [u32],
    pub kv_buf: &'b mut [T],
    pub scratch_buf: &'b mut [T],
    pub kv_used_tokens: &'b mut [usize],
    pub quant_packed: &'b mut [u8],
    pub quant_scales: &'b mut [T],
    pub sink: Option<&'a mut dyn crate::engine::infer::inference::TokenSink>,
}

pub fn run_lm<'a, 'b, T: crate::base::math::Float + 'static>(
    cfg: &crate::graph::lm::LmConfig<T>,
    inputs: LmRunInputs<'a, T>,
    bufs: LmRunBufs<'a, 'b, T>,
) -> Result<(f64, usize), RnnApiError> {
    let LmRunInputs {
        weights_buf,
        prompt_ids,
        max_new_tokens,
        sampling,
        prompt_already_in_cache,
        history,
        stop_ids,
        quant,
        lora,
    } = inputs;
    let LmRunBufs {
        out_ids,
        kv_buf,
        scratch_buf,
        kv_used_tokens,
        quant_packed,
        quant_scales,
        sink,
    } = bufs;
    use crate::graph::lm::{LmBlockWeights, BlockTWeights, BlockQWeights, LmQuant, lm_weights_read_into};
    use crate::engine::infer::inference::{ForwardScratch, generate, GenerateArgs};
    use crate::graph::attn::kv_cache::KvCacheView;
    use core::mem::MaybeUninit;

    cfg.validate().map_err(|_| RnnApiError::BadBytes)?;
    if cfg.num_layers > MAX_LAYERS { return Err(RnnApiError::BadBytes); }
    if kv_used_tokens.len() < cfg.num_layers { return Err(RnnApiError::CapacityTooSmall); }

    let kv_count = cfg.num_layers * 2 * cfg.context_len * cfg.kv_h();
    if kv_buf.len() < kv_count { return Err(RnnApiError::CapacityTooSmall); }
    if scratch_buf.len() < ForwardScratch::total_f32_count(cfg) { return Err(RnnApiError::CapacityTooSmall); }

    if let Some(mut shared) = crate::base::tensor::SharedTensor::alloc([1, 1, 1, 1, kv_count.min(16)]) {
        if shared.ref_count() != 1 { return Err(RnnApiError::CapacityTooSmall); }
        let n = shared.as_slice().len().min(shared.as_mut_slice().len());
        if n == 0 { return Err(RnnApiError::CapacityTooSmall); }
    }

    let head_dim = cfg.head_dim().map_err(|_| RnnApiError::BadBytes)?;
    let kv_stride = cfg.context_len * cfg.kv_h();

    let block_bytes = cfg.num_layers * core::mem::size_of::<LmBlockWeights<'static, T>>();
    let blocks_ptr = crate::engine::runtime::hardware::mmap_shared_anon(block_bytes);
    if blocks_ptr.is_null() { return Err(RnnApiError::CapacityTooSmall); }

    unsafe {
        let raw = blocks_ptr as *mut LmBlockWeights<'static, T>;
        for i in 0..cfg.num_layers {
            raw.add(i).write(LmBlockWeights {
                attn_norm_gamma: &[], wq: &[], wk: &[], wv: &[], wo: &[],
                ffn_norm_gamma: &[], w_gate: &[], w_up: &[], w_down: &[],
                moe: None,
            });
        }
    }

    let blocks_slice: &mut [LmBlockWeights<'_, T>] = unsafe {
        core::mem::transmute::<&mut [LmBlockWeights<'static, T>], &mut [LmBlockWeights<'_, T>]>(
            core::slice::from_raw_parts_mut(blocks_ptr as *mut LmBlockWeights<'static, T>, cfg.num_layers)
        )
    };

    let mut weights = match lm_weights_read_into(cfg, weights_buf, blocks_slice) {
        Ok(w) => w,
        Err(_) => {
            crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
            return Err(RnnApiError::BadBytes);
        }
    };

    let mut t_blocks_ptr: *mut u8 = core::ptr::null_mut();
    let mut t_blocks_bytes = 0usize;
    let mut q_blocks_ptr: *mut u8 = core::ptr::null_mut();
    let mut q_blocks_bytes = 0usize;
    if quant == LmQuant::Ternary {
        if quant_packed.len() < cfg.ternary_packed_u8_count()
            || quant_scales.len() < cfg.ternary_scales_f32_count()
        {
            crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
            return Err(RnnApiError::CapacityTooSmall);
        }
        t_blocks_bytes = cfg.num_layers * core::mem::size_of::<BlockTWeights<'static, T>>();
        t_blocks_ptr = crate::engine::runtime::hardware::mmap_shared_anon(t_blocks_bytes);
        if t_blocks_ptr.is_null() {
            crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
            return Err(RnnApiError::CapacityTooSmall);
        }
        let t_blocks_slice: &mut [BlockTWeights<'_, T>] = unsafe {
            core::mem::transmute::<&mut [BlockTWeights<'static, T>], &mut [BlockTWeights<'_, T>]>(
                core::slice::from_raw_parts_mut(
                    t_blocks_ptr as *mut BlockTWeights<'static, T>,
                    cfg.num_layers,
                ),
            )
        };
        match crate::graph::lm::lm_weights_quantize_ternary_into(
            cfg, &weights, quant_packed, quant_scales, t_blocks_slice,
        ) {
            Ok(t) => weights.t = Some(t),
            Err(_) => {
                crate::engine::runtime::hardware::munmap(t_blocks_ptr, t_blocks_bytes);
                crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
                return Err(RnnApiError::Model);
            }
        }
    } else if quant == LmQuant::Int8 {
        if quant_packed.len() < cfg.int8_i8_count()
            || quant_scales.len() < cfg.int8_scales_f32_count()
        {
            crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
            return Err(RnnApiError::CapacityTooSmall);
        }
        q_blocks_bytes = cfg.num_layers * core::mem::size_of::<BlockQWeights<'static, T>>();
        q_blocks_ptr = crate::engine::runtime::hardware::mmap_shared_anon(q_blocks_bytes);
        if q_blocks_ptr.is_null() {
            crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
            return Err(RnnApiError::CapacityTooSmall);
        }
        let q_blocks_slice: &mut [BlockQWeights<'_, T>] = unsafe {
            core::mem::transmute::<&mut [BlockQWeights<'static, T>], &mut [BlockQWeights<'_, T>]>(
                core::slice::from_raw_parts_mut(
                    q_blocks_ptr as *mut BlockQWeights<'static, T>,
                    cfg.num_layers,
                ),
            )
        };
        let quant_code: &mut [i8] = unsafe {
            core::slice::from_raw_parts_mut(
                quant_packed.as_mut_ptr() as *mut i8,
                quant_packed.len(),
            )
        };
        match crate::graph::lm::lm_weights_quantize_int8_into(
            cfg, &weights, quant_code, quant_scales, q_blocks_slice,
        ) {
            Ok(q) => weights.q = Some(q),
            Err(_) => {
                crate::engine::runtime::hardware::munmap(q_blocks_ptr, q_blocks_bytes);
                crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
                return Err(RnnApiError::Model);
            }
        }
    }

    let h = cfg.hidden_size;
    let f = cfg.ffw_size;
    let c = cfg.context_len;
    let v = cfg.vocab_size;
    let kv_h = cfg.kv_h();
    let (hidden, rest) = scratch_buf.split_at_mut(h);
    let (norm_buf, rest) = rest.split_at_mut(h);
    let (qkv, rest) = rest.split_at_mut(h + 2 * kv_h);
    let (wo_out, rest) = rest.split_at_mut(h);
    let (scores, rest) = rest.split_at_mut(c);
    let (attn_head, rest) = rest.split_at_mut(h);
    let (ffn_gate, rest) = rest.split_at_mut(f);
    let (ffn_up, rest) = rest.split_at_mut(f);
    let (ffn_out, rest) = rest.split_at_mut(h);
    let (logits, rest) = rest.split_at_mut(v);
    let hh = cfg.head_hidden;
    let (head_z1, rest) = rest.split_at_mut(hh);
    let (head_a1, _) = rest.split_at_mut(hh);
    let mut sc = ForwardScratch { hidden, norm_buf, qkv, wo_out, scores, attn_head, ffn_gate, ffn_up, ffn_out, logits, head_z1, head_a1, moe_aux_loss: 0.0 };

    let mut kv_uninit: [MaybeUninit<KvCacheView<'_, T>>; MAX_LAYERS] =
        [const { MaybeUninit::uninit() }; MAX_LAYERS];
    let mut kv_remaining = &mut kv_buf[..];
    for (i, slot) in kv_uninit.iter_mut().take(cfg.num_layers).enumerate() {
        let (key_part, rest) = kv_remaining.split_at_mut(kv_stride);
        let (val_part, next) = rest.split_at_mut(kv_stride);
        kv_remaining = next;
        slot.write(KvCacheView {
            key: key_part,
            value: val_part,
            max_tokens: cfg.context_len,
            head_dim,
            num_heads: cfg.num_kv_heads,
            used_tokens: kv_used_tokens[i],
        });
    }
    let kv_layers: &mut [KvCacheView<'_, T>] = unsafe {
        core::slice::from_raw_parts_mut(kv_uninit[0].as_mut_ptr(), cfg.num_layers)
    };

    let start_position = kv_used_tokens[0];
    let mut args = GenerateArgs::new(prompt_ids, max_new_tokens, sampling);
    args.start_position = start_position;
    args.prompt_already_in_cache = prompt_already_in_cache;
    args.history = history;
    args.stop_ids = stop_ids;
    args.sink = sink;

    let pool = crate::engine::train::trainer::par_pool::par_pool_launch();

    let mut lora_blocks_ptr: *mut u8 = core::ptr::null_mut();
    let mut lora_blocks_bytes = 0usize;
    let mut adapted_ptr: *mut u8 = core::ptr::null_mut();
    let mut adapted_bytes = 0usize;

    let gen_result = if let Some(spec) = lora {
        match build_lora_blocks(cfg, &spec) {
            Ok((bp, bb, ap, ab, blocks, mut adapted)) => {
                lora_blocks_ptr = bp;
                lora_blocks_bytes = bb;
                adapted_ptr = ap;
                adapted_bytes = ab;
                crate::engine::infer::inference::generate_lora(
                    crate::engine::infer::inference::LoraModel { cfg, w: &weights, loras: blocks, adapted: &mut adapted },
                    args,
                    kv_layers,
                    &mut sc,
                    out_ids,
                )
            }
            Err(e) => {
                if !t_blocks_ptr.is_null() {
                    crate::engine::runtime::hardware::munmap(t_blocks_ptr, t_blocks_bytes);
                }
                if !q_blocks_ptr.is_null() {
                    crate::engine::runtime::hardware::munmap(q_blocks_ptr, q_blocks_bytes);
                }
                crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
                return Err(e);
            }
        }
    } else {
        generate(cfg, &weights, args, kv_layers, &mut sc, out_ids)
    };

    let outcome = match gen_result {
        Ok(o) => o,
        Err(_) => {
            if !adapted_ptr.is_null() {
                crate::engine::runtime::hardware::munmap(adapted_ptr, adapted_bytes);
            }
            if !lora_blocks_ptr.is_null() {
                crate::engine::runtime::hardware::munmap(lora_blocks_ptr, lora_blocks_bytes);
            }
            if !t_blocks_ptr.is_null() {
                crate::engine::runtime::hardware::munmap(t_blocks_ptr, t_blocks_bytes);
            }
            if !q_blocks_ptr.is_null() {
                crate::engine::runtime::hardware::munmap(q_blocks_ptr, q_blocks_bytes);
            }
            crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
            return Err(RnnApiError::Model);
        }
    };
    let generation_halted_early = outcome.completed_early();
    let next_position = outcome.next_position();
    for i in 0..cfg.num_layers {
        let mut final_len = kv_layers[i].len_tokens();
        if generation_halted_early {
            final_len = final_len.max(next_position);
        }
        kv_used_tokens[i] = final_len;
    }
    if !adapted_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(adapted_ptr, adapted_bytes);
    }
    if !lora_blocks_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(lora_blocks_ptr, lora_blocks_bytes);
    }
    if !t_blocks_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(t_blocks_ptr, t_blocks_bytes);
    }
    if !q_blocks_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(q_blocks_ptr, q_blocks_bytes);
    }
    crate::engine::runtime::hardware::munmap(blocks_ptr, block_bytes);
    crate::engine::train::trainer::par_pool::par_pool_destroy(&pool);
    Ok((0.0, outcome.generated))
}

type LoraBlocksHandle<'p, 'a, T> = (
    *mut u8,
    usize,
    *mut u8,
    usize,
    &'p mut [crate::graph::lm::LmBlockLora<'a, T>],
    crate::graph::lm::AdaptedWeightBuf<'p, T>,
);

fn build_lora_blocks<'p, 'a, T: crate::base::math::Float + 'static>(
    cfg: &crate::graph::lm::LmConfig<T>,
    spec: &LoraSpec<'a, T>,
) -> Result<LoraBlocksHandle<'p, 'a, T>, RnnApiError> {
    use crate::graph::lm::{AdaptedWeightBuf, LmBlockLora};

    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let r = spec.rank;
    if r == 0 {
        return Err(RnnApiError::BadBytes);
    }
    let per_layer = 6 * r * h + 2 * kv_h * r;
    if spec.pack.len() < cfg.num_layers * per_layer {
        return Err(RnnApiError::CapacityTooSmall);
    }

    let blocks_bytes = cfg.num_layers * core::mem::size_of::<LmBlockLora<'static, T>>();
    let blocks_ptr = crate::engine::runtime::hardware::mmap_shared_anon(blocks_bytes);
    if blocks_ptr.is_null() {
        return Err(RnnApiError::CapacityTooSmall);
    }

    let elem = core::mem::size_of::<T>();
    let adapted_bytes = 4 * h * h * elem;
    let adapted_ptr = crate::engine::runtime::hardware::mmap_shared_anon(adapted_bytes);
    if adapted_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(blocks_ptr, blocks_bytes);
        return Err(RnnApiError::CapacityTooSmall);
    }

    let pack = spec.pack;
    let raw = blocks_ptr as *mut LmBlockLora<'a, T>;
    for layer in 0..cfg.num_layers {
        let mut off = layer * per_layer;
        let take = |n: usize, off: &mut usize| -> &'a [T] {
            let s = &pack[*off..*off + n];
            *off += n;
            s
        };
        let wq_a = take(r * h, &mut off);
        let wq_b = take(h * r, &mut off);
        let wk_a = take(r * h, &mut off);
        let wk_b = take(kv_h * r, &mut off);
        let wv_a = take(r * h, &mut off);
        let wv_b = take(kv_h * r, &mut off);
        let wo_a = take(r * h, &mut off);
        let wo_b = take(h * r, &mut off);
        unsafe {
            raw.add(layer).write(LmBlockLora {
                wq_a: Some(wq_a),
                wq_b: Some(wq_b),
                wk_a: Some(wk_a),
                wk_b: Some(wk_b),
                wv_a: Some(wv_a),
                wv_b: Some(wv_b),
                wo_a: Some(wo_a),
                wo_b: Some(wo_b),
                rank: r,
                alpha: spec.alpha,
            });
        }
    }
    let blocks: &mut [LmBlockLora<'a, T>] =
        unsafe { core::slice::from_raw_parts_mut(raw, cfg.num_layers) };

    let adapted_base = adapted_ptr as *mut T;
    let (wq, wk, wv, wo) = unsafe {
        (
            core::slice::from_raw_parts_mut(adapted_base, h * h),
            core::slice::from_raw_parts_mut(adapted_base.add(h * h), h * h),
            core::slice::from_raw_parts_mut(adapted_base.add(2 * h * h), h * h),
            core::slice::from_raw_parts_mut(adapted_base.add(3 * h * h), h * h),
        )
    };
    let adapted = AdaptedWeightBuf { wq, wk, wv, wo };

    Ok((blocks_ptr, blocks_bytes, adapted_ptr, adapted_bytes, blocks, adapted))
}
