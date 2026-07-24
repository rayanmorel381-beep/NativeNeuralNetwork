use crate::graph::attn::kv_cache::KvCacheView;
use crate::base::tensor::CacheTensor;
use crate::text::sampling::{
    argmax_sample, sample_from_cumulative, softmax_temperature, top_k_mask,
    top_p_cutoff,
};
use crate::graph::lm::{LmConfig, LmError, LmWeights, SamplingConfig};
use crate::graph::lm::{AdaptedWeightBuf, LmBlockLora};
use crate::graph::lm::Lcg;
use crate::base::math::Float;
use super::scratch::ForwardScratch;
use super::transformer_forward::{transformer_forward, transformer_forward_lora, LoraWeights};

pub trait TokenSink {
    fn on_token(&mut self, id: u32) -> bool;
}

pub struct GenerateArgs<'a, T: Float> {
    pub prompt_ids: &'a [u32],
    pub max_new_tokens: usize,
    pub sampling: &'a SamplingConfig<T>,
    pub start_position: usize,
    pub prompt_already_in_cache: bool,
    pub history: &'a [u32],
    pub stop_ids: &'a [u32],
    pub sink: Option<&'a mut dyn TokenSink>,
}

impl<'a, T: Float> GenerateArgs<'a, T> {
    pub fn new(prompt_ids: &'a [u32], max_new_tokens: usize, sampling: &'a SamplingConfig<T>) -> Self {
        Self {
            prompt_ids,
            max_new_tokens,
            sampling,
            start_position: 0,
            prompt_already_in_cache: false,
            history: &[],
            stop_ids: &[],
            sink: None,
        }
    }
}

pub struct GenerateOutcome {
    pub generated: usize,
    pub final_position: usize,
    pub stopped_by_eos: bool,
    pub stopped_by_sink: bool,
}

impl GenerateOutcome {
    pub fn completed_early(&self) -> bool {
        self.stopped_by_eos || self.stopped_by_sink
    }

    pub fn next_position(&self) -> usize {
        self.final_position
    }
}

pub fn generate<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    args: GenerateArgs<'_, T>,
    kv_layers: &mut [KvCacheView<T>],
    sc: &mut ForwardScratch<T>,
    out_ids: &mut [u32],
) -> Result<GenerateOutcome, LmError> {
    if args.sampling.beam_width >= 2 && !args.prompt_already_in_cache {
        return generate_beam(cfg, w, args, sc, out_ids);
    }
    let GenerateArgs {
        prompt_ids,
        max_new_tokens,
        sampling,
        start_position,
        prompt_already_in_cache,
        history,
        stop_ids,
        mut sink,
    } = args;

    if out_ids.len() < max_new_tokens {
        return Err(LmError::BufferTooSmall);
    }

    if !sc.validate_sizes(cfg) {
        return Err(LmError::ScratchTooSmall);
    }

    let v = cfg.vocab_size;
    if prompt_ids.is_empty() {
        return Err(LmError::ShapeMismatch);
    }

    if prompt_already_in_cache {
        if let Some(first) = kv_layers.first() {
            if first.is_empty() {
                return Err(LmError::ShapeMismatch);
            }
        }
    } else {
        for kv in kv_layers.iter_mut() {
            kv.clear();
        }
    }

    let mut rng = Lcg::new(sampling.rng_seed);
    let mut position = start_position;

    if !prompt_already_in_cache {
        if prompt_ids.len() >= 2 {
            super::prefill::prefill_prompt(cfg, w, prompt_ids, kv_layers, sc.hidden)?;
            position = start_position + prompt_ids.len();
        } else {
            for &tok in prompt_ids {
                transformer_forward(cfg, w, tok, position, kv_layers, sc)?;
                position += 1;
            }
        }
    }

    let cache_capacity = kv_layers
        .first()
        .map(|kv| kv.remaining_tokens())
        .unwrap_or(0);

    if let Some(first) = kv_layers.first_mut() {
        if let Some(ct) = CacheTensor::<T>::from_raw(
            first.key.as_mut_ptr(),
            first.max_tokens.max(1),
            first.num_heads,
            first.head_dim,
        ) {
            let used = ct.used_tokens();
            let stride = ct.stride();
            if stride != first.num_heads * first.head_dim
                || used > first.max_tokens
                || ct.as_slice().len() < used * stride
            {
                return Err(LmError::KvCapacityExceeded);
            }
        }
    }
    let mut staging_val = T::ZERO;
    if let Some(mut ct_stage) = CacheTensor::<T>::from_raw(
        &mut staging_val as *mut T, 1, 1, 1
    ) {
        let kv_tok = [T::ZERO];
        if ct_stage.push(&kv_tok) {
            let staged_len = ct_stage.as_slice().len();
            if staged_len == 0 { return Err(LmError::ScratchTooSmall); }
        }
    }

    let mut generated = 0usize;
    let mut last_token = *prompt_ids.last().ok_or(LmError::ShapeMismatch)?;
    let mut stopped_by_eos = false;
    let mut stopped_by_sink = false;

    while generated < max_new_tokens && generated < cache_capacity {
        transformer_forward(cfg, w, last_token, position, kv_layers, sc)?;
        position += 1;

        apply_penalties(sc.logits, v, sampling, history, prompt_ids, &out_ids[..generated]);

        let next_id = sample_next(sc.logits, v, sampling, &mut rng)?;
        out_ids[generated] = next_id;
        generated += 1;

        if next_id == sampling.eos_id || stop_ids.contains(&next_id) {
            stopped_by_eos = true;
            if let Some(s) = sink.as_deref_mut() {
                s.on_token(next_id);
            }
            break;
        }

        if let Some(s) = sink.as_deref_mut() {
            if !s.on_token(next_id) {
                stopped_by_sink = true;
                break;
            }
        }

        last_token = next_id;
    }

    Ok(GenerateOutcome { generated, final_position: position, stopped_by_eos, stopped_by_sink })
}

pub struct LoraModel<'a, T: Float> {
    pub cfg: &'a LmConfig<T>,
    pub w: &'a LmWeights<'a, T>,
    pub loras: &'a [LmBlockLora<'a, T>],
    pub adapted: &'a mut AdaptedWeightBuf<'a, T>,
}

pub fn generate_lora<T: Float>(
    model: LoraModel<'_, T>,
    args: GenerateArgs<'_, T>,
    kv_layers: &mut [KvCacheView<T>],
    sc: &mut ForwardScratch<T>,
    out_ids: &mut [u32],
) -> Result<GenerateOutcome, LmError> {
    let LoraModel { cfg, w, loras, adapted } = model;
    let mut lw = LoraWeights { loras, adapted };
    let GenerateArgs {
        prompt_ids,
        max_new_tokens,
        sampling,
        start_position,
        prompt_already_in_cache,
        history,
        stop_ids,
        mut sink,
    } = args;

    if out_ids.len() < max_new_tokens {
        return Err(LmError::BufferTooSmall);
    }

    if !sc.validate_sizes(cfg) {
        return Err(LmError::ScratchTooSmall);
    }

    let v = cfg.vocab_size;
    if prompt_ids.is_empty() {
        return Err(LmError::ShapeMismatch);
    }

    let mut rng = Lcg::new(sampling.rng_seed);
    let mut position = start_position;

    if !prompt_already_in_cache {
        for &tok in prompt_ids {
            transformer_forward_lora(cfg, w, &mut lw, tok, position, kv_layers, sc)?;
            position += 1;
        }
    }

    let mut generated = 0usize;
    let mut last_token = *prompt_ids.last().ok_or(LmError::ShapeMismatch)?;
    let mut stopped_by_eos = false;
    let mut stopped_by_sink = false;

    while generated < max_new_tokens {
        transformer_forward_lora(cfg, w, &mut lw, last_token, position, kv_layers, sc)?;
        position += 1;

        apply_penalties(sc.logits, v, sampling, history, prompt_ids, &out_ids[..generated]);

        let next_id = sample_next(sc.logits, v, sampling, &mut rng)?;
        out_ids[generated] = next_id;
        generated += 1;

        if next_id == sampling.eos_id || stop_ids.contains(&next_id) {
            stopped_by_eos = true;
            if let Some(s) = sink.as_deref_mut() {
                s.on_token(next_id);
            }
            break;
        }

        if let Some(s) = sink.as_deref_mut() {
            if !s.on_token(next_id) {
                stopped_by_sink = true;
                break;
            }
        }

        last_token = next_id;
    }

    Ok(GenerateOutcome { generated, final_position: position, stopped_by_eos, stopped_by_sink })
}

fn apply_penalties<T: Float>(
    logits: &mut [T],
    vocab_size: usize,
    cfg: &SamplingConfig<T>,
    history: &[u32],
    prompt: &[u32],
    produced: &[u32],
) {
    let rep = cfg.repetition_penalty;
    let pres = cfg.presence_penalty;
    let freq = cfg.frequency_penalty;
    let rep_active = rep.is_finite() && rep > T::ZERO && (rep - T::ONE).abs() > T::EPSILON;
    let pres_active = pres.abs() > T::ZERO;
    let freq_active = freq.abs() > T::ZERO;
    if !rep_active && !pres_active && !freq_active {
        return;
    }

    let logits = &mut logits[..vocab_size];
    let win = cfg.repetition_window;

    let total: usize = history.len() + prompt.len() + produced.len();
    let start_drop = if win == 0 || total <= win { 0 } else { total - win };

    let mut idx = 0usize;
    let mut visit = |id: u32, slot: &mut usize| {
        let cur = *slot;
        *slot += 1;
        if cur < start_drop {
            return;
        }
        let i = id as usize;
        if i >= logits.len() {
            return;
        }
        if rep_active {
            let l = logits[i];
            if l > T::ZERO {
                logits[i] = l / rep;
            } else {
                logits[i] = l * rep;
            }
        }
        if pres_active {
            logits[i] -= pres;
        }
        if freq_active {
            logits[i] -= freq;
        }
    };

    for &id in history {
        visit(id, &mut idx);
    }
    for &id in prompt {
        visit(id, &mut idx);
    }
    for &id in produced {
        visit(id, &mut idx);
    }
}

fn sample_next<T: Float>(
    logits: &mut [T],
    vocab_size: usize,
    cfg: &SamplingConfig<T>,
    rng: &mut Lcg,
) -> Result<u32, LmError> {
    let logits = &mut logits[..vocab_size];

    if cfg.temperature <= T::ZERO || !cfg.temperature.is_finite() {
        return argmax_sample(logits)
            .map(|id| id as u32)
            .map_err(|_| LmError::SamplingError);
    }

    softmax_temperature(logits, cfg.temperature).map_err(|_| LmError::SamplingError)?;

    if cfg.top_k > 0 {
        top_k_mask(logits, cfg.top_k, T::ZERO).map_err(|_| LmError::SamplingError)?;
        let mut sum = T::ZERO;
        for v in logits.iter() {
            sum += *v;
        }
        if sum > T::ZERO {
            let inv = T::ONE / sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    if cfg.top_p < T::ONE {
        let cutoff = top_p_cutoff(logits, cfg.top_p)
            .map_err(|_| LmError::SamplingError)?;
        for v in logits[cutoff..].iter_mut() {
            *v = T::ZERO;
        }
        let mut sum = T::ZERO;
        for v in logits.iter() {
            sum += *v;
        }
        if sum > T::ZERO {
            let inv = T::ONE / sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    let threshold = rng.next_uniform::<T>();
    sample_from_cumulative(logits, threshold)
        .map(|id| id as u32)
        .map_err(|_| LmError::SamplingError)
}

fn beam_log_softmax<T: Float>(logits: &mut [T]) -> bool {
    match T::DTYPE_TAG {
        0 => {
            let l = unsafe { core::slice::from_raw_parts_mut(logits.as_mut_ptr().cast::<f32>(), logits.len()) };
            crate::text::beam_search::log_softmax_in_place_f32(l)
        }
        1 => {
            let l = unsafe { core::slice::from_raw_parts_mut(logits.as_mut_ptr().cast::<f64>(), logits.len()) };
            crate::text::beam_search::log_softmax_in_place_f64(l)
        }
        _ => false,
    }
}

pub fn generate_beam<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    args: GenerateArgs<'_, T>,
    sc: &mut ForwardScratch<T>,
    out_ids: &mut [u32],
) -> Result<GenerateOutcome, LmError> {
    use crate::engine::rnn_flow::MAX_LAYERS;
    use core::mem::MaybeUninit;

    let GenerateArgs {
        prompt_ids,
        max_new_tokens,
        sampling,
        history,
        stop_ids,
        ..
    } = args;

    let v = cfg.vocab_size;
    let num_layers = cfg.num_layers;
    if num_layers > MAX_LAYERS {
        return Err(LmError::ShapeMismatch);
    }
    if sc.logits.len() < v {
        return Err(LmError::ScratchTooSmall);
    }
    if prompt_ids.is_empty() {
        return Err(LmError::ShapeMismatch);
    }
    if out_ids.len() < max_new_tokens {
        return Err(LmError::BufferTooSmall);
    }

    let head_dim = cfg.head_dim()?;
    let num_kv_heads = cfg.num_kv_heads;
    let kv_stride = cfg.context_len * cfg.kv_h();
    let kv_count = num_layers * 2 * kv_stride;
    let context_len = cfg.context_len;

    let prompt_len = prompt_ids.len();
    let room = context_len.saturating_sub(prompt_len);
    let mn = max_new_tokens.min(room);
    if mn == 0 {
        return Ok(GenerateOutcome {
            generated: 0,
            final_position: prompt_len,
            stopped_by_eos: false,
            stopped_by_sink: false,
        });
    }


    let wb = sampling.beam_width;
    let szt = core::mem::size_of::<T>();
    let align8 = |x: usize| (x + 7) & !7;

    let mut off = 0usize;
    let o_kvc = off; off += align8(wb * kv_count * szt);
    let o_kvn = off; off += align8(wb * kv_count * szt);
    let o_sqc = off; off += align8(wb * mn * 4);
    let o_sqn = off; off += align8(wb * mn * 4);
    let o_fsq = off; off += align8(wb * mn * 4);
    let o_scc = off; off += align8(wb * 8);
    let o_scn = off; off += align8(wb * 8);
    let o_clp = off; off += align8(wb * wb * 8);
    let o_fsc = off; off += align8(wb * 8);
    let o_cpar = off; off += align8(wb * wb * 8);
    let o_topw = off; off += align8(wb * 8);
    let o_sel = off; off += align8(wb * wb * 8);
    let o_flen = off; off += align8(wb * 8);
    let o_ctok = off; off += align8(wb * wb * 4);
    let o_lc = off; off += align8(wb * 4);
    let o_ln = off; off += align8(wb * 4);
    let o_csc = off; off += align8(wb * wb * 4);
    let o_tmp = off; off += align8(v * 4);
    let total = off;

    let base = crate::engine::runtime::hardware::mmap_shared_anon(total);
    if base.is_null() {
        return Err(LmError::BufferTooSmall);
    }

    let mut run = || -> Result<GenerateOutcome, LmError> {
        let mut kv_cur = unsafe { base.add(o_kvc) as *mut T };
        let mut kv_nxt = unsafe { base.add(o_kvn) as *mut T };
        let mut seq_cur = unsafe { core::slice::from_raw_parts_mut(base.add(o_sqc) as *mut u32, wb * mn) };
        let mut seq_nxt = unsafe { core::slice::from_raw_parts_mut(base.add(o_sqn) as *mut u32, wb * mn) };
        let fin_seq = unsafe { core::slice::from_raw_parts_mut(base.add(o_fsq) as *mut u32, wb * mn) };
        let mut score_cur = unsafe { core::slice::from_raw_parts_mut(base.add(o_scc) as *mut f64, wb) };
        let mut score_nxt = unsafe { core::slice::from_raw_parts_mut(base.add(o_scn) as *mut f64, wb) };
        let cand_lp = unsafe { core::slice::from_raw_parts_mut(base.add(o_clp) as *mut f64, wb * wb) };
        let fin_score = unsafe { core::slice::from_raw_parts_mut(base.add(o_fsc) as *mut f64, wb) };
        let cand_par = unsafe { core::slice::from_raw_parts_mut(base.add(o_cpar) as *mut usize, wb * wb) };
        let topw = unsafe { core::slice::from_raw_parts_mut(base.add(o_topw) as *mut usize, wb) };
        let sel = unsafe { core::slice::from_raw_parts_mut(base.add(o_sel) as *mut usize, wb * wb) };
        let fin_len = unsafe { core::slice::from_raw_parts_mut(base.add(o_flen) as *mut usize, wb) };
        let cand_tok = unsafe { core::slice::from_raw_parts_mut(base.add(o_ctok) as *mut u32, wb * wb) };
        let mut last_cur = unsafe { core::slice::from_raw_parts_mut(base.add(o_lc) as *mut u32, wb) };
        let mut last_nxt = unsafe { core::slice::from_raw_parts_mut(base.add(o_ln) as *mut u32, wb) };
        let cand_sc = unsafe { core::slice::from_raw_parts_mut(base.add(o_csc) as *mut f32, wb * wb) };
        let tmp = unsafe { core::slice::from_raw_parts_mut(base.add(o_tmp) as *mut f32, v) };

        let forward_beam = |kv_base: *mut T, beam: usize, last_token: u32, position: usize, used: usize, sc: &mut ForwardScratch<T>| -> Result<(), LmError> {
            let mut views: [MaybeUninit<KvCacheView<T>>; MAX_LAYERS] = [const { MaybeUninit::uninit() }; MAX_LAYERS];
            for (l, slot) in views.iter_mut().take(num_layers).enumerate() {
                let kbase = unsafe { kv_base.add(beam * kv_count + l * 2 * kv_stride) };
                let key = unsafe { core::slice::from_raw_parts_mut(kbase, kv_stride) };
                let value = unsafe { core::slice::from_raw_parts_mut(kbase.add(kv_stride), kv_stride) };
                slot.write(KvCacheView {
                    key, value,
                    max_tokens: context_len,
                    head_dim,
                    num_heads: num_kv_heads,
                    used_tokens: used,
                });
            }
            let vslice = unsafe { core::slice::from_raw_parts_mut(views[0].as_mut_ptr(), num_layers) };
            transformer_forward(cfg, w, last_token, position, vslice, sc)
        };

        let mut cur_len = prompt_len - 1;
        for (i, &tok) in prompt_ids.iter().take(prompt_len - 1).enumerate() {
            forward_beam(kv_cur, 0, tok, i, i, sc)?;
        }

        let mut active = 1usize;
        score_cur[0] = 0.0;
        last_cur[0] = prompt_ids[prompt_len - 1];

        let mut t = 0usize;
        let mut fcount = 0usize;

        while t < mn && active > 0 {
            let total_cand = active * wb;
            for c in cand_sc[..total_cand].iter_mut() {
                *c = f32::NEG_INFINITY;
            }

            for bm in 0..active {
                forward_beam(kv_cur, bm, last_cur[bm], cur_len, cur_len, sc)?;
                apply_penalties(sc.logits, v, sampling, history, prompt_ids, &seq_cur[bm * mn..bm * mn + t]);
                if !beam_log_softmax(&mut sc.logits[..v]) {
                    return Err(LmError::SamplingError);
                }
                for (i, slot) in tmp[..v].iter_mut().enumerate() {
                    *slot = sc.logits[i].to_f64() as f32;
                }
                let filled = crate::text::beam_search::select_top_beams(&tmp[..v], wb, topw)
                    .map_err(|_| LmError::SamplingError)?;
                for (j, &topw_token) in topw.iter().take(filled).enumerate() {
                    let tok = topw_token as u32;
                    let c = bm * wb + j;
                    let lp = sc.logits[tok as usize].to_f64();
                    cand_lp[c] = score_cur[bm] + lp;
                    cand_par[c] = bm;
                    cand_tok[c] = tok;
                    cand_sc[c] = cand_lp[c] as f32;
                }
            }

            let chosen = crate::text::beam_search::select_top_beams(&cand_sc[..total_cand], wb, sel)
                .map_err(|_| LmError::SamplingError)?;

            let mut na = 0usize;
            for &c in sel.iter().take(chosen) {
                if !cand_sc[c].is_finite() {
                    continue;
                }
                let par = cand_par[c];
                let tok = cand_tok[c];
                let lp = cand_lp[c];
                let is_fin = tok == sampling.eos_id || stop_ids.contains(&tok);
                if is_fin {
                    if fcount < wb {
                        fin_seq[fcount * mn..fcount * mn + t]
                            .copy_from_slice(&seq_cur[par * mn..par * mn + t]);
                        fin_seq[fcount * mn + t] = tok;
                        fin_len[fcount] = t + 1;
                        fin_score[fcount] = lp / ((t + 1) as f64);
                        fcount += 1;
                    }
                } else {
                    unsafe {
                        core::ptr::copy_nonoverlapping(
                            kv_cur.add(par * kv_count),
                            kv_nxt.add(na * kv_count),
                            kv_count,
                        );
                    }
                    seq_nxt[na * mn..na * mn + t].copy_from_slice(&seq_cur[par * mn..par * mn + t]);
                    seq_nxt[na * mn + t] = tok;
                    score_nxt[na] = lp;
                    last_nxt[na] = tok;
                    na += 1;
                }
            }

            core::mem::swap(&mut kv_cur, &mut kv_nxt);
            core::mem::swap(&mut seq_cur, &mut seq_nxt);
            core::mem::swap(&mut score_cur, &mut score_nxt);
            core::mem::swap(&mut last_cur, &mut last_nxt);
            active = na;
            cur_len += 1;
            t += 1;

            if fcount >= wb {
                break;
            }
        }

        let mut best_norm = f64::NEG_INFINITY;
        let mut best_seq_off = 0usize;
        let mut best_len = 0usize;
        let mut best_eos = false;
        let mut from_active = false;
        let mut found = false;

        for f in 0..fcount {
            if fin_score[f] > best_norm {
                best_norm = fin_score[f];
                best_seq_off = f * mn;
                best_len = fin_len[f];
                best_eos = true;
                from_active = false;
                found = true;
            }
        }
        if active > 0 && t > 0 {
            let nrm = score_cur[0] / (t as f64);
            if nrm > best_norm {
                best_seq_off = 0;
                best_len = t;
                best_eos = false;
                from_active = true;
                found = true;
            }
        }

        if !found {
            return Ok(GenerateOutcome {
                generated: 0,
                final_position: cur_len,
                stopped_by_eos: false,
                stopped_by_sink: false,
            });
        }

        if from_active {
            out_ids[..best_len].copy_from_slice(&seq_cur[best_seq_off..best_seq_off + best_len]);
        } else {
            out_ids[..best_len].copy_from_slice(&fin_seq[best_seq_off..best_seq_off + best_len]);
        }

        Ok(GenerateOutcome {
            generated: best_len,
            final_position: cur_len,
            stopped_by_eos: best_eos,
            stopped_by_sink: false,
        })
    };

    let outcome = run();
    crate::engine::runtime::hardware::munmap(base, total);
    outcome
}

#[cfg(test)]
mod beam_tests {
    extern crate std;
    use super::*;
    use crate::graph::lm::{lm_weights_param_count, lm_weights_read_into, LmBlockWeights};
    use std::vec;
    use std::vec::Vec;

    fn tiny_cfg() -> LmConfig<f32> {
        LmConfig {
            vocab_size: 19,
            context_len: 12,
            hidden_size: 8,
            ffw_size: 16,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            rope_theta: 10_000.0,
            head_hidden: 8,
        }
    }

    fn make_params(n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        let mut state: u64 = 0x1357_9bdf_2468_ace0;
        for slot in out.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((state >> 33) as u32) as f32 / (u32::MAX as f32);
            *slot = r - 0.5;
        }
        out
    }

    #[test]
    fn beam_search_generates_valid_tokens() {
        let cfg = tiny_cfg();
        let n = lm_weights_param_count(&cfg);
        let params = make_params(n);

        let mut blocks: Vec<LmBlockWeights<'_, f32>> = (0..cfg.num_layers)
            .map(|_| LmBlockWeights {
                attn_norm_gamma: &[], wq: &[], wk: &[], wv: &[], wo: &[],
                ffn_norm_gamma: &[], w_gate: &[], w_up: &[], w_down: &[],
                moe: None,
            })
            .collect();
        let weights = lm_weights_read_into(&cfg, &params, &mut blocks).unwrap();

        let mut scratch = vec![0.0f32; ForwardScratch::<f32>::total_f32_count(&cfg)];
        let h = cfg.hidden_size;
        let f = cfg.ffw_size;
        let c = cfg.context_len;
        let v = cfg.vocab_size;
        let kv_h = cfg.kv_h();
        let (hidden, rest) = scratch.split_at_mut(h);
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
        let mut sc = ForwardScratch {
            hidden, norm_buf, qkv, wo_out, scores, attn_head, ffn_gate, ffn_up, ffn_out, logits, head_z1, head_a1, moe_aux_loss: 0.0,
        };

        let mut sampling = SamplingConfig::<f32>::greedy(18);
        sampling.beam_width = 3;

        let prompt: [u32; 3] = [1, 5, 3];
        let max_new = 5usize;
        let mut out_ids = vec![0u32; max_new];

        let args = GenerateArgs::new(&prompt, max_new, &sampling);
        let outcome = generate_beam(&cfg, &weights, args, &mut sc, &mut out_ids).unwrap();

        assert!(outcome.generated <= max_new);
        assert!(outcome.generated > 0, "beam search produced no tokens");
        for &id in &out_ids[..outcome.generated] {
            assert!((id as usize) < cfg.vocab_size, "token {} out of vocab", id);
        }
    }
}
