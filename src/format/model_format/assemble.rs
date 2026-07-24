use crate::security::crypto::encrypt_rnn;
use crate::format::model_config::{model_config, BuildConfigError, Precision};
use crate::format::model_format::model_format;
use crate::api::rnn_api::core_api::RnnApiError;

fn map_engine_error(err: crate::engine::RnnFlowError) -> RnnApiError {
    match err {
        crate::engine::RnnFlowError::InvalidTopology => RnnApiError::InvalidTopology,
        crate::engine::RnnFlowError::CapacityTooSmall => RnnApiError::CapacityTooSmall,
        crate::engine::RnnFlowError::BadBytes => RnnApiError::BadBytes,
        crate::engine::RnnFlowError::Model => RnnApiError::Model,
    }
}

fn map_config_error(e: BuildConfigError) -> RnnApiError {
    match e {
        BuildConfigError::InvalidTopology => RnnApiError::InvalidTopology,
        BuildConfigError::Layer => RnnApiError::Layer,
        BuildConfigError::Model => RnnApiError::Model,
    }
}

pub(crate) fn build_with_precision(
    model_name: &str,
    topology: &[usize],
    precision: &Precision<'_>,
    benchmark_override: Option<&[u8]>,
    buf: &mut [u8],
) -> Result<usize, RnnApiError> {
    let half = buf.len() / 2;
    let (out, scratch) = buf.split_at_mut(half);
    let scratch_half = scratch.len() / 2;
    let (rmd1_area, metadata_area) = scratch.split_at_mut(scratch_half);

    let rmd1_used = model_config(topology, precision, rmd1_area)
        .map_err(map_config_error)?;
    let used = model_format(
        model_name,
        topology,
        &rmd1_area[..rmd1_used],
        benchmark_override,
        precision,
        metadata_area,
        out,
    )
    .map_err(map_engine_error)?;
    let enc_len = encrypt_rnn(out, used);
    Ok(enc_len)
}
