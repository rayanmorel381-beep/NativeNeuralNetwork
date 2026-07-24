use super::{run_lm, LmRunBufs, LmRunInputs, LoraSpec};
use crate::graph::lm::{
    build_chat_prompt, chat_stop_ids, lm_container_read, lm_weights_param_count, ChatMessage,
    ChatRole, ChatSpecialTokens, LmConfig, LmQuant, SamplingConfig,
};
use crate::base::math::Float;
use crate::api::rnn_api::core_api::RnnApiError;
use crate::format::model_config::LmLoraInput;
use crate::text::tokenizer::{
    decode_ids_bpe, encode_with_bpe, parse_merges_blob, parse_vocab_blob, MergePair, Vocab,
};

pub struct LmChatInputs<'a> {
    pub contents: &'a [u8],
    pub content_lens: &'a [usize],
    pub roles: &'a [u32],
    pub specials: ChatSpecialTokens,
    pub max_new_tokens: usize,
    pub quant: LmQuant,
    pub lora: Option<LmLoraInput<'a>>,
}

pub struct LmChatBufs<'a> {
    pub out_text: &'a mut [u8],
}

fn role_from_u32(raw: u32) -> ChatRole {
    match raw {
        0 => ChatRole::System,
        1 => ChatRole::User,
        2 => ChatRole::Assistant,
        _ => ChatRole::Tool,
    }
}

fn align_up(x: usize, a: usize) -> usize {
    (x + a - 1) & !(a - 1)
}

pub fn run_lm_chat<T: Float + 'static>(
    container: &[u8],
    inputs: LmChatInputs,
    mut bufs: LmChatBufs,
) -> Result<(f64, usize), RnnApiError> {
    let view: crate::graph::lm::LmContainerView<'_, T> =
        lm_container_read::<T>(container).map_err(|_| RnnApiError::BadBytes)?;
    let cfg = view.cfg;
    let vocab_blob = view.vocab.ok_or(RnnApiError::BadBytes)?;

    let vocab_size = cfg.vocab_size;
    if vocab_size == 0 || inputs.content_lens.len() != inputs.roles.len() {
        return Err(RnnApiError::BadBytes);
    }
    let num_messages = inputs.content_lens.len();
    let param_count = lm_weights_param_count(&cfg);
    let esz = core::mem::size_of::<T>();
    if view.weights.len() != param_count * esz {
        return Err(RnnApiError::BadBytes);
    }

    let total_content: usize = inputs.content_lens.iter().sum();
    if total_content != inputs.contents.len() {
        return Err(RnnApiError::BadBytes);
    }
    let max_single = inputs.content_lens.iter().copied().max().unwrap_or(0).max(1);
    let encoded_cap = total_content.max(1);
    let prompt_cap = total_content + num_messages * 4 + 4;
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

    let entry_size = core::mem::size_of::<(&[u8], u32)>();
    let merge_size = core::mem::size_of::<MergePair>();
    let msg_size = core::mem::size_of::<ChatMessage<'_>>();

    let entries_off = 0usize;
    let merges_off = align_up(entries_off + vocab_size * entry_size, 8);
    let weights_off = align_up(merges_off + max_merges * merge_size, 8);
    let encoded_off = align_up(weights_off + param_count * esz, 8);
    let scratch_ids_off = align_up(encoded_off + encoded_cap * 4, 8);
    let messages_off = align_up(scratch_ids_off + max_single * 4, 8);
    let prompt_off = align_up(messages_off + num_messages.max(1) * msg_size, 8);
    let out_ids_off = align_up(prompt_off + prompt_cap * 4, 8);
    let kv_off = align_up(out_ids_off + out_ids_count * 4, 8);
    let scratch_off = align_up(kv_off + kv_count * esz, 8);
    let kv_used_off = align_up(scratch_off + scratch_count * esz, 8);
    let qp_off = align_up(kv_used_off + kv_used_count * core::mem::size_of::<usize>(), 8);
    let qs_off = align_up(qp_off + qp_count, 8);
    let total = qs_off + (qs_count * esz).max(1);

    let base = crate::engine::runtime::hardware::mmap_shared_anon(total);
    if base.is_null() {
        return Err(RnnApiError::CapacityTooSmall);
    }

    unsafe {
        core::ptr::copy_nonoverlapping(view.weights.as_ptr(), base.add(weights_off), param_count * esz);
    }

    let layout = ChatLayout {
        base,
        entries_off,
        merges_off,
        weights_off,
        encoded_off,
        scratch_ids_off,
        messages_off,
        prompt_off,
        out_ids_off,
        kv_off,
        scratch_off,
        kv_used_off,
        qp_off,
        qs_off,
        vocab_size,
        max_merges,
        param_count,
        encoded_cap,
        max_single,
        num_messages,
        prompt_cap,
        out_ids_count,
        kv_count,
        scratch_count,
        kv_used_count,
        qp_count,
        qs_count,
    };

    let result = run_chat_inner::<T>(&cfg, vocab_blob, view.merges, &inputs, &mut bufs, layout);

    crate::engine::runtime::hardware::munmap(base, total);
    result.map(|text_len| (0.0, text_len))
}

struct ChatLayout {
    base: *mut u8,
    entries_off: usize,
    merges_off: usize,
    weights_off: usize,
    encoded_off: usize,
    scratch_ids_off: usize,
    messages_off: usize,
    prompt_off: usize,
    out_ids_off: usize,
    kv_off: usize,
    scratch_off: usize,
    kv_used_off: usize,
    qp_off: usize,
    qs_off: usize,
    vocab_size: usize,
    max_merges: usize,
    param_count: usize,
    encoded_cap: usize,
    max_single: usize,
    num_messages: usize,
    prompt_cap: usize,
    out_ids_count: usize,
    kv_count: usize,
    scratch_count: usize,
    kv_used_count: usize,
    qp_count: usize,
    qs_count: usize,
}

fn run_chat_inner<'c, T: Float + 'static>(
    cfg: &LmConfig<T>,
    vocab_blob: &'c [u8],
    merges_blob: Option<&'c [u8]>,
    inputs: &LmChatInputs,
    bufs: &mut LmChatBufs,
    layout: ChatLayout,
) -> Result<usize, RnnApiError> {
    let base = layout.base;

    let empty: &[u8] = &[];
    let entries_raw = unsafe { base.add(layout.entries_off) } as *mut (&[u8], u32);
    for i in 0..layout.vocab_size {
        unsafe {
            entries_raw.add(i).write((empty, 0u32));
        }
    }
    let entries: &mut [(&'c [u8], u32)] =
        unsafe { core::slice::from_raw_parts_mut(entries_raw, layout.vocab_size) };
    let merges_store: &mut [MergePair] = unsafe {
        core::slice::from_raw_parts_mut(base.add(layout.merges_off) as *mut MergePair, layout.max_merges)
    };
    let weights: &[T] =
        unsafe { core::slice::from_raw_parts(base.add(layout.weights_off) as *const T, layout.param_count) };
    let encoded: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.encoded_off) as *mut u32, layout.encoded_cap) };
    let encode_scratch: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.scratch_ids_off) as *mut u32, layout.max_single) };
    let prompt_ids: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.prompt_off) as *mut u32, layout.prompt_cap) };
    let out_ids: &mut [u32] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.out_ids_off) as *mut u32, layout.out_ids_count) };
    let kv_buf: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.kv_off) as *mut T, layout.kv_count) };
    let scratch_buf: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.scratch_off) as *mut T, layout.scratch_count) };
    let kv_used_tokens: &mut [usize] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.kv_used_off) as *mut usize, layout.kv_used_count) };
    let quant_packed: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.qp_off), layout.qp_count) };
    let quant_scales: &mut [T] =
        unsafe { core::slice::from_raw_parts_mut(base.add(layout.qs_off) as *mut T, layout.qs_count) };

    let (nb, unk_id) = parse_vocab_blob(vocab_blob, entries).map_err(|_| RnnApiError::BadBytes)?;
    let vocab = Vocab::new(&entries[..nb], unk_id).map_err(|_| RnnApiError::BadBytes)?;
    let merges: &[MergePair] = match merges_blob {
        Some(mb) => {
            let n = parse_merges_blob(mb, merges_store).map_err(|_| RnnApiError::BadBytes)?;
            &merges_store[..n]
        }
        None => &[],
    };

    let messages_raw = unsafe { base.add(layout.messages_off) } as *mut ChatMessage<'_>;
    let mut content_off = 0usize;
    let mut encoded_off = 0usize;
    for (i, &clen) in inputs.content_lens.iter().enumerate() {
        let content = &inputs.contents[content_off..content_off + clen];
        let written = encode_with_bpe(
            content,
            &vocab,
            merges,
            &mut encoded[encoded_off..],
            encode_scratch,
        )
        .map_err(|_| RnnApiError::CapacityTooSmall)?;
        let ids: &[u32] = unsafe {
            core::slice::from_raw_parts((encoded.as_ptr()).add(encoded_off), written)
        };
        unsafe {
            messages_raw.add(i).write(ChatMessage {
                role: role_from_u32(inputs.roles[i]),
                content_ids: ids,
            });
        }
        content_off += clen;
        encoded_off += written;
    }
    let messages: &[ChatMessage<'_>] =
        unsafe { core::slice::from_raw_parts(messages_raw, layout.num_messages) };

    let prompt_len = build_chat_prompt(messages, &inputs.specials, true, prompt_ids)
        .map_err(|_| RnnApiError::CapacityTooSmall)?;

    let stops = chat_stop_ids(&inputs.specials);
    let sampling = SamplingConfig::<T>::default_sampling(inputs.specials.eos.unwrap_or(u32::MAX));

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
            stop_ids: &stops,
            quant: inputs.quant,
            lora,
        },
        LmRunBufs {
            out_ids,
            kv_buf,
            scratch_buf,
            kv_used_tokens,
            quant_packed,
            quant_scales,
            sink: None,
        },
    )?;

    decode_ids_bpe(&out_ids[..generated], merges, bufs.out_text)
        .map_err(|_| RnnApiError::CapacityTooSmall)
}
