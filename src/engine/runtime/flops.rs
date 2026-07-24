use super::{RuntimeError, RuntimeProfile};
use crate::format::model_config::{ConfigError, TransformerConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeFlopsEstimate {
    pub prefill_flops: u128,
    pub decode_token_flops: u128,
    pub total_flops: u128,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThroughputEstimate {
    pub flops_per_token: u128,
    pub estimated_tokens_per_second: f32,
}

pub fn estimate_runtime_flops(
    config: &TransformerConfig,
    profile: RuntimeProfile,
) -> Result<RuntimeFlopsEstimate, RuntimeError> {
    config.validate().map_err(|_| RuntimeError::InvalidConfig)?;
    if !profile.validate() {
        return Err(RuntimeError::InvalidProfile);
    }

    let h = config.hidden_size as u128;
    let f = config.ffw_size as u128;
    let l = config.num_layers as u128;
    let b = profile.batch_size as u128;
    let s = profile.sequence_len as u128;

    let qkv = h
        .checked_mul(h)
        .and_then(|x| x.checked_mul(6))
        .ok_or(RuntimeError::Overflow)?;
    let o_proj = h
        .checked_mul(h)
        .and_then(|x| x.checked_mul(2))
        .ok_or(RuntimeError::Overflow)?;
    let ff = h
        .checked_mul(f)
        .and_then(|x| x.checked_mul(4))
        .ok_or(RuntimeError::Overflow)?;

    let per_token = qkv
        .checked_add(o_proj)
        .and_then(|x| x.checked_add(ff))
        .ok_or(RuntimeError::Overflow)?;

    let per_token_attention = s
        .checked_mul(h)
        .and_then(|x| x.checked_mul(4))
        .ok_or(RuntimeError::Overflow)?;

    let per_token_per_layer = per_token
        .checked_add(per_token_attention)
        .ok_or(RuntimeError::Overflow)?;

    let tokens = profile.token_count().ok_or(RuntimeError::Overflow)? as u128;
    let prefill_flops = tokens
        .checked_mul(l)
        .and_then(|x| x.checked_mul(per_token_per_layer))
        .ok_or(RuntimeError::Overflow)?;

    let decode_attention = s
        .checked_mul(h)
        .and_then(|x| x.checked_mul(2))
        .ok_or(RuntimeError::Overflow)?;
    let decode_per_token_per_layer = per_token
        .checked_add(decode_attention)
        .ok_or(RuntimeError::Overflow)?;
    let decode_token_flops = b
        .checked_mul(l)
        .and_then(|x| x.checked_mul(decode_per_token_per_layer))
        .ok_or(RuntimeError::Overflow)?;

    let total_flops = prefill_flops
        .checked_add(decode_token_flops)
        .ok_or(RuntimeError::Overflow)?;

    Ok(RuntimeFlopsEstimate {
        prefill_flops,
        decode_token_flops,
        total_flops,
    })
}

pub fn estimate_tokens_per_second(
    config: &TransformerConfig,
    profile: RuntimeProfile,
    sustained_flops_per_second: u128,
) -> Result<ThroughputEstimate, RuntimeError> {
    config.validate().map_err(|e| match e {
        ConfigError::Invalid => RuntimeError::InvalidConfig,
        ConfigError::Overflow => RuntimeError::Overflow,
    })?;

    if !profile.validate() || sustained_flops_per_second == 0 {
        return Err(RuntimeError::InvalidProfile);
    }

    let flops = estimate_runtime_flops(config, profile)?;
    let flops_per_token = flops.decode_token_flops;
    if flops_per_token == 0 {
        return Err(RuntimeError::InvalidProfile);
    }

    let tps = sustained_flops_per_second as f32 / flops_per_token as f32;
    Ok(ThroughputEstimate {
        flops_per_token,
        estimated_tokens_per_second: tps,
    })
}
