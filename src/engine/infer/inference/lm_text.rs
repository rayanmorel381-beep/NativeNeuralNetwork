use super::{run_lm, LmRunBufs, LmRunInputs, LoraSpec};
use crate::graph::lm::{lm_weights_param_count, lm_container_read, LmConfig, LmQuant, SamplingConfig};
use crate::base::math::Float;
use crate::api::rnn_api::core_api::RnnApiError;
use crate::format::model_config::LmLoraInput;
use crate::text::tokenizer::{
    decode_ids_bpe, encode_with_bpe, parse_merges_blob, parse_vocab_blob, MergePair, Vocab,
};

pub struct LmTextInputs<'a> {
    pub prompt_text: &'a [u8],
    pub max_new_tokens: usize,
    pub quant: LmQuant,
    pub lora: Option<LmLoraInput<'a>>,
}

pub struct LmTextBufs<'a> {
    pub out_text: &'a mut [u8],
}

struct RunBufs<'s, T> {
    out_ids: &'s mut [u32],
    kv_buf: &'s mut [T],
    scratch_buf: &'s mut [T],
    kv_used_tokens: &'s mut [usize],
    quant_packed: &'s mut [u8],
    quant_scales: &'s mut [T],
}

struct TokScratch<'c, 's> {
    entries: &'s mut [(&'c [u8], u32)],
    merges_store: &'s mut [MergePair],
    prompt_ids: &'s mut [u32],
    encode_scratch: &'s mut [u32],
}

pub fn run_lm_text<T: Float + 'static>(
    container: &[u8],
    inputs: LmTextInputs,
    mut bufs: LmTextBufs,
) -> Result<(f64, usize), RnnApiError> {
    let view: crate::graph::lm::LmContainerView<'_, T> =
        lm_container_read::<T>(container).map_err(|_| RnnApiError::BadBytes)?;
    let cfg = view.cfg;
    let vocab_blob = view.vocab.ok_or(RnnApiError::BadBytes)?;

    let vocab_size = cfg.vocab_size;
    if vocab_size == 0 {
        return Err(RnnApiError::BadBytes);
    }
    let param_count = lm_weights_param_count(&cfg);
    let esz = core::mem::size_of::<T>();
    if view.weights.len() != param_count * esz {
        return Err(RnnApiError::BadBytes);
    }

    let plen_cap = inputs.prompt_text.len().max(1);
    let max_merges = view.merges.map(|m| m.len() / 12 + 1).unwrap_or(0).max(1);

    let out_ids_count = inputs.max_new_tokens.max(1);
    let kv_count = cfg.kv_cache_f32_count();
    let scratch_count = cfg.forward_scratch_f32_count();
    let kv_used_count = cfg.num_layers.max(1);
    let (qp_count, qs_count) = match inputs.quant {
        LmQuant::None => (0usize, 0usize),
        LmQuant::Ternary => (
            cfg.ternary_packed_u8_count(),
            cfg.ternary_scales_f32_count(),
        ),
        LmQuant::Int8 => (cfg.int8_i8_count(), cfg.int8_scales_f32_count()),
    };

    let sizes = [
        vocab_size * core::mem::size_of::<(&[u8], u32)>(),
        max_merges * core::mem::size_of::<MergePair>(),
        plen_cap * 2 * 4,
        param_count * esz,
        out_ids_count * 4,
        (kv_count * esz).max(1),
        (scratch_count * esz).max(1),
        kv_used_count * core::mem::size_of::<usize>(),
        qp_count.max(1),
        (qs_count * esz).max(1),
    ];

    let mut ptrs = [core::ptr::null_mut::<u8>(); 10];
    for i in 0..sizes.len() {
        let p = crate::engine::runtime::hardware::mmap_shared_anon(sizes[i]);
        if p.is_null() {
            for j in 0..i {
                crate::engine::runtime::hardware::munmap(ptrs[j], sizes[j]);
            }
            return Err(RnnApiError::CapacityTooSmall);
        }
        ptrs[i] = p;
    }

    unsafe {
        core::ptr::copy_nonoverlapping(view.weights.as_ptr(), ptrs[3], param_count * esz);
    }

    let empty: &[u8] = &[];
    let entries_raw = ptrs[0] as *mut (&[u8], u32);
    for i in 0..vocab_size {
        unsafe {
            entries_raw.add(i).write((empty, 0u32));
        }
    }

    let entries: &mut [(&[u8], u32)] =
        unsafe { core::slice::from_raw_parts_mut(entries_raw, vocab_size) };
    let merges_store: &mut [MergePair] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[1] as *mut MergePair, max_merges) };
    let ids: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[2] as *mut u32, plen_cap * 2) };
    let (prompt_ids, encode_scratch) = ids.split_at_mut(plen_cap);
    let weights: &[T] =
        unsafe { core::slice::from_raw_parts(ptrs[3] as *const T, param_count) };
    let out_ids: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[4] as *mut u32, out_ids_count) };
    let kv_buf: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[5] as *mut T, kv_count) };
    let scratch_buf: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[6] as *mut T, scratch_count) };
    let kv_used_tokens: &mut [usize] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[7] as *mut usize, kv_used_count) };
    let quant_packed: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[8], qp_count) };
    let quant_scales: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(ptrs[9] as *mut T, qs_count) };

    let result = run_text_inner(
        &cfg,
        vocab_blob,
        view.merges,
        weights,
        TokScratch {
            entries,
            merges_store,
            prompt_ids,
            encode_scratch,
        },
        RunBufs {
            out_ids,
            kv_buf,
            scratch_buf,
            kv_used_tokens,
            quant_packed,
            quant_scales,
        },
        &inputs,
        &mut bufs,
    );

    for i in 0..sizes.len() {
        crate::engine::runtime::hardware::munmap(ptrs[i], sizes[i]);
    }

    result.map(|text_len| (0.0, text_len))
}

#[allow(clippy::too_many_arguments)]
fn run_text_inner<'c, T: Float + 'static>(
    cfg: &LmConfig<T>,
    vocab_blob: &'c [u8],
    merges_blob: Option<&'c [u8]>,
    weights: &[T],
    scratch: TokScratch<'c, '_>,
    run_bufs: RunBufs<'_, T>,
    inputs: &LmTextInputs,
    bufs: &mut LmTextBufs,
) -> Result<usize, RnnApiError> {
    let TokScratch {
        entries,
        merges_store,
        prompt_ids,
        encode_scratch,
    } = scratch;
    let (nb, unk_id) = parse_vocab_blob(vocab_blob, entries).map_err(|_| RnnApiError::BadBytes)?;
    let vocab = Vocab::new(&entries[..nb], unk_id).map_err(|_| RnnApiError::BadBytes)?;

    let merges: &[MergePair] = match merges_blob {
        Some(mb) => {
            let n = parse_merges_blob(mb, merges_store).map_err(|_| RnnApiError::BadBytes)?;
            &merges_store[..n]
        }
        None => &[],
    };

    let prompt_len = encode_with_bpe(
        inputs.prompt_text,
        &vocab,
        merges,
        prompt_ids,
        encode_scratch,
    )
    .map_err(|_| RnnApiError::CapacityTooSmall)?;

    let sampling = SamplingConfig::<T>::default_sampling(u32::MAX);

    let lora = match &inputs.lora {
        Some(li) => {
            let esz = core::mem::size_of::<T>();
            if li.pack.is_empty()
                || li.rank == 0
                || li.pack.len() % esz != 0
                || (li.pack.as_ptr() as usize) % core::mem::align_of::<T>() != 0
            {
                return Err(RnnApiError::BadBytes);
            }
            let pack: &[T] = unsafe {
                core::slice::from_raw_parts(li.pack.as_ptr() as *const T, li.pack.len() / esz)
            };
            Some(LoraSpec { pack, rank: li.rank, alpha: T::from_f64(li.alpha) })
        }
        None => None,
    };

    let (_, generated) = run_lm::<T>(
        cfg,
        LmRunInputs {
            weights_buf: weights,
            prompt_ids: &prompt_ids[..prompt_len],
            max_new_tokens: inputs.max_new_tokens,
            sampling: &sampling,
            prompt_already_in_cache: false,
            history: &[],
            stop_ids: &[],
            quant: inputs.quant,
            lora,
        },
        LmRunBufs {
            out_ids: run_bufs.out_ids,
            kv_buf: run_bufs.kv_buf,
            scratch_buf: run_bufs.scratch_buf,
            kv_used_tokens: run_bufs.kv_used_tokens,
            quant_packed: run_bufs.quant_packed,
            quant_scales: run_bufs.quant_scales,
            sink: None,
        },
    )?;

    decode_ids_bpe(&run_bufs.out_ids[..generated], merges, bufs.out_text)
        .map_err(|_| RnnApiError::CapacityTooSmall)
}
