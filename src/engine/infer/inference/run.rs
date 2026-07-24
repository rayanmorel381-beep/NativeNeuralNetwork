use crate::api::rnn_api::core_api::{ensure_rnn_initialized, map_engine_error, RnnApiError};

pub(crate) fn run_inference(
    bytes: &mut [u8],
    config: &mut crate::format::model_config::ModelConfig,
) -> Result<(f64, usize), RnnApiError> {
    if let crate::format::model_config::Precision::LmF32 {
        cfg, weights_buf, prompt_ids, max_new_tokens, sampling, out_ids,
        kv_buf, scratch_buf, kv_used_tokens, prompt_already_in_cache,
        history, stop_ids, quant, quant_packed, quant_scales, sink,
    } = &mut config.precision
    {
        return crate::engine::infer::inference::run_lm(
            *cfg,
            crate::engine::infer::inference::LmRunInputs {
                weights_buf,
                prompt_ids,
                max_new_tokens: *max_new_tokens,
                sampling,
                prompt_already_in_cache: *prompt_already_in_cache,
                history,
                stop_ids,
                quant: *quant,
                lora: None,
            },
            crate::engine::infer::inference::LmRunBufs {
                out_ids,
                kv_buf,
                scratch_buf,
                kv_used_tokens,
                quant_packed,
                quant_scales,
                sink: sink.take(),
            },
        );
    }

    if let crate::format::model_config::Precision::LmF64 {
        cfg, weights_buf, prompt_ids, max_new_tokens, sampling, out_ids,
        kv_buf, scratch_buf, kv_used_tokens, prompt_already_in_cache,
        history, stop_ids, quant, quant_packed, quant_scales, sink,
    } = &mut config.precision
    {
        return crate::engine::infer::inference::run_lm(
            *cfg,
            crate::engine::infer::inference::LmRunInputs {
                weights_buf,
                prompt_ids,
                max_new_tokens: *max_new_tokens,
                sampling,
                prompt_already_in_cache: *prompt_already_in_cache,
                history,
                stop_ids,
                quant: *quant,
                lora: None,
            },
            crate::engine::infer::inference::LmRunBufs {
                out_ids,
                kv_buf,
                scratch_buf,
                kv_used_tokens,
                quant_packed,
                quant_scales,
                sink: sink.take(),
            },
        );
    }

    if let crate::format::model_config::Precision::LmText {
        prompt_text, out_text, max_new_tokens, quant, lora,
    } = &mut config.precision
    {
        let enc_len = bytes.len();
        let plain_len = crate::security::crypto::decrypt_rnn(bytes, enc_len);
        let was_encrypted = plain_len != enc_len;
        let dtype = crate::graph::lm::lm_weights_dtype(&bytes[..plain_len])
            .map_err(|_| RnnApiError::BadBytes)?;
        let inputs = crate::engine::infer::inference::LmTextInputs {
            prompt_text,
            max_new_tokens: *max_new_tokens,
            quant: *quant,
            lora: *lora,
        };
        let result = match dtype {
            0 => crate::engine::infer::inference::run_lm_text::<f32>(
                &bytes[..plain_len],
                inputs,
                crate::engine::infer::inference::LmTextBufs { out_text },
            ),
            1 => crate::engine::infer::inference::run_lm_text::<f64>(
                &bytes[..plain_len],
                inputs,
                crate::engine::infer::inference::LmTextBufs { out_text },
            ),
            _ => return Err(RnnApiError::BadBytes),
        };
        if was_encrypted {
            crate::security::crypto::encrypt_rnn(bytes, plain_len);
        }
        return result;
    }

    if let crate::format::model_config::Precision::LmChat {
        contents, content_lens, roles, specials, out_text, max_new_tokens, quant, lora,
    } = &mut config.precision
    {
        let enc_len = bytes.len();
        let plain_len = crate::security::crypto::decrypt_rnn(bytes, enc_len);
        let was_encrypted = plain_len != enc_len;
        let dtype = crate::graph::lm::lm_weights_dtype(&bytes[..plain_len])
            .map_err(|_| RnnApiError::BadBytes)?;
        let inputs = crate::engine::infer::inference::LmChatInputs {
            contents,
            content_lens,
            roles,
            specials: *specials,
            max_new_tokens: *max_new_tokens,
            quant: *quant,
            lora: *lora,
        };
        let result = match dtype {
            0 => crate::engine::infer::inference::run_lm_chat::<f32>(
                &bytes[..plain_len],
                inputs,
                crate::engine::infer::inference::LmChatBufs { out_text },
            ),
            1 => crate::engine::infer::inference::run_lm_chat::<f64>(
                &bytes[..plain_len],
                inputs,
                crate::engine::infer::inference::LmChatBufs { out_text },
            ),
            _ => return Err(RnnApiError::BadBytes),
        };
        if was_encrypted {
            crate::security::crypto::encrypt_rnn(bytes, plain_len);
        }
        return result;
    }

    ensure_rnn_initialized(bytes)?;
    crate::engine::runtime::ensure_hardware_init();

    if let Some((evolve_idx, iteration_seed)) = config.evolve_out {
        let enc_len = config.encrypted_data_len.unwrap_or(bytes.len());
        return crate::engine::train::trainer::execute_evolve(bytes, enc_len, evolve_idx, iteration_seed)
            .map_err(map_engine_error);
    }

    crate::engine::run_mlp(bytes, config)
}
