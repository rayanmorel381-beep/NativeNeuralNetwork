use super::profile::RuntimeProfile;
use crate::format::model_config::{ConfigError, TransformerConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeEstimate {
    pub parameter_bytes: usize,
    pub activation_bytes: usize,
    pub kv_cache_bytes: usize,
    pub total_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidProfile,
    Overflow,
    InvalidConfig,
}

pub fn estimate_runtime_memory(
    config: &TransformerConfig,
    profile: RuntimeProfile,
) -> Result<RuntimeEstimate, RuntimeError> {
    config.validate().map_err(|_| RuntimeError::InvalidConfig)?;

    if !profile.validate() {
        return Err(RuntimeError::InvalidProfile);
    }

    let params = config.approximate_parameter_count().map_err(|e| match e {
        ConfigError::Invalid => RuntimeError::InvalidConfig,
        ConfigError::Overflow => RuntimeError::Overflow,
    })?;

    let parameter_bytes = params
        .checked_mul(profile.bytes_per_param)
        .ok_or(RuntimeError::Overflow)?;
    let tokens = profile
        .batch_size
        .checked_mul(profile.sequence_len)
        .ok_or(RuntimeError::Overflow)?;
    let activation_elems = tokens
        .checked_mul(config.hidden_size)
        .and_then(|x| x.checked_mul(config.num_layers))
        .ok_or(RuntimeError::Overflow)?;
    let activation_bytes = activation_elems
        .checked_mul(profile.bytes_per_activation)
        .ok_or(RuntimeError::Overflow)?;

    let kv_elems = tokens
        .checked_mul(config.hidden_size)
        .and_then(|x| x.checked_mul(config.num_layers))
        .and_then(|x| x.checked_mul(2))
        .ok_or(RuntimeError::Overflow)?;
    let kv_cache_bytes = kv_elems
        .checked_mul(profile.bytes_per_kv)
        .ok_or(RuntimeError::Overflow)?;

    let total_bytes = parameter_bytes
        .checked_add(activation_bytes)
        .and_then(|x| x.checked_add(kv_cache_bytes))
        .ok_or(RuntimeError::Overflow)?;

    Ok(RuntimeEstimate {
        parameter_bytes,
        activation_bytes,
        kv_cache_bytes,
        total_bytes,
    })
}
