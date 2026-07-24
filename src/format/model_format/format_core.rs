use crate::graph::net::layers::{LayerDesc, LayerSpec};
use crate::base::math::Float;
use crate::format::model_format::header::{write_header, RMD1_HEADER_SIZE};
use crate::format::model_format::payload_bounds::has_full_payload;

const HEADER_SIZE: usize = RMD1_HEADER_SIZE;
const LAYER_META_SIZE: usize = 4 + 4 + 4 + 4 + 1 + 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelFormatError {
    BadHeader,
    CapacityTooSmall,
    InvalidLayer,
}

fn encoded_size_bytes(
    scalar_bytes: usize,
    layer_count: usize,
    weights_len: usize,
    biases_len: usize,
) -> Option<usize> {
    let layers_bytes = layer_count.checked_mul(LAYER_META_SIZE)?;
    let weights_bytes = weights_len.checked_mul(scalar_bytes)?;
    let biases_bytes = biases_len.checked_mul(scalar_bytes)?;
    HEADER_SIZE
        .checked_add(layers_bytes)?
        .checked_add(weights_bytes)?
        .checked_add(biases_bytes)
}

pub fn scalar_bytes_for_dtype(dtype: u8) -> Option<usize> {
    if dtype == f32::DTYPE_TAG {
        Some(f32::BYTE_SIZE)
    } else if dtype == f64::DTYPE_TAG {
        Some(f64::BYTE_SIZE)
    } else {
        None
    }
}

pub fn encoded_size(
    scalar_bytes: usize,
    layer_count: usize,
    weights_len: usize,
    biases_len: usize,
) -> Option<usize> {
    encoded_size_bytes(scalar_bytes, layer_count, weights_len, biases_len)
}

pub fn encode_model<T: Float>(
    layers: &[LayerSpec],
    weights: &[T],
    biases: &[T],
    out: &mut [u8],
) -> Result<usize, ModelFormatError> {
    let needed = encoded_size_bytes(T::BYTE_SIZE, layers.len(), weights.len(), biases.len())
        .ok_or(ModelFormatError::BadHeader)?;
    if out.len() < needed {
        return Err(ModelFormatError::CapacityTooSmall);
    }

    write_header(out, T::DTYPE_TAG, layers.len(), weights.len(), biases.len())
        .ok_or(ModelFormatError::BadHeader)?;

    let mut cursor = HEADER_SIZE;
    for layer in layers {
        let dense = match layer {
            LayerSpec::Dense(v) => *v,
        };
        validate(&dense, weights.len(), biases.len())?;

        out[cursor..cursor + 4].copy_from_slice(
            &u32::try_from(dense.input_size)
                .map_err(|_| ModelFormatError::BadHeader)?
                .to_le_bytes(),
        );
        cursor += 4;
        out[cursor..cursor + 4].copy_from_slice(
            &u32::try_from(dense.output_size)
                .map_err(|_| ModelFormatError::BadHeader)?
                .to_le_bytes(),
        );
        cursor += 4;
        out[cursor..cursor + 4].copy_from_slice(
            &u32::try_from(dense.weight_offset)
                .map_err(|_| ModelFormatError::BadHeader)?
                .to_le_bytes(),
        );
        cursor += 4;
        out[cursor..cursor + 4].copy_from_slice(
            &u32::try_from(dense.bias_offset)
                .map_err(|_| ModelFormatError::BadHeader)?
                .to_le_bytes(),
        );
        cursor += 4;
        out[cursor] = dense.activation.to_u8();
        cursor += 1;
        out[cursor..cursor + 3].copy_from_slice(&[0u8; 3]);
        cursor += 3;
    }

    for &w in weights {
        w.write_le_bytes(&mut out[cursor..cursor + T::BYTE_SIZE]);
        cursor += T::BYTE_SIZE;
    }
    for &b in biases {
        b.write_le_bytes(&mut out[cursor..cursor + T::BYTE_SIZE]);
        cursor += T::BYTE_SIZE;
    }

    if !has_full_payload(&out[..cursor]) {
        return Err(ModelFormatError::BadHeader);
    }

    Ok(cursor)
}

fn validate(
    layer: &LayerDesc,
    weights_len: usize,
    biases_len: usize,
) -> Result<(), ModelFormatError> {
    if layer.input_size == 0 || layer.output_size == 0 {
        return Err(ModelFormatError::InvalidLayer);
    }
    let w_len = layer.weight_len().ok_or(ModelFormatError::InvalidLayer)?;
    let w_end = layer
        .weight_offset
        .checked_add(w_len)
        .ok_or(ModelFormatError::InvalidLayer)?;
    let b_end = layer
        .bias_offset
        .checked_add(layer.output_size)
        .ok_or(ModelFormatError::InvalidLayer)?;
    if w_end > weights_len || b_end > biases_len {
        return Err(ModelFormatError::InvalidLayer);
    }
    Ok(())
}

pub(crate) fn model_format(
    _model_name: &str,
    topology: &[usize],
    rmd1: &[u8],
    benchmark_override: Option<&[u8]>,
    precision: &crate::format::model_config::Precision,
    metadata_scratch: &mut [u8],
    out: &mut [u8],
) -> Result<usize, crate::engine::RnnFlowError> {
    use super::container::{
        build_container_from_rmd1, build_container_from_rmd1_with_benchmark,
        build_container_from_rmd1_with_conv_spec, RuntimeInput,
    };
    match precision {
        crate::format::model_config::Precision::F32 { runtime_input, conv_spec, .. } => {
            if let Some(conv_spec_bytes) = *conv_spec {
                let req = super::container::ContainerBuildRequest {
                    model_name: _model_name,
                    topology,
                    precision_str: "f32",
                    rmd1_bytes: rmd1,
                    benchmark_override,
                    runtime_input: runtime_input.map(RuntimeInput::F32),
                    conv_spec: Some(conv_spec_bytes),
                };
                build_container_from_rmd1_with_conv_spec(
                    &req,
                    metadata_scratch,
                    out,
                )
            } else if let Some(benchmark_blob) = benchmark_override {
                let req = super::container::ContainerBuildRequest {
                    model_name: _model_name,
                    topology,
                    precision_str: "f32",
                    rmd1_bytes: rmd1,
                    benchmark_override: Some(benchmark_blob),
                    runtime_input: runtime_input.map(RuntimeInput::F32),
                    conv_spec: None,
                };
                build_container_from_rmd1_with_benchmark(
                    &req,
                    benchmark_blob,
                    metadata_scratch,
                    out,
                )
            } else {
                let req = super::container::ContainerBuildRequest {
                    model_name: _model_name,
                    topology,
                    precision_str: "f32",
                    rmd1_bytes: rmd1,
                    benchmark_override: None,
                    runtime_input: runtime_input.map(RuntimeInput::F32),
                    conv_spec: None,
                };
                build_container_from_rmd1(
                    &req,
                    metadata_scratch,
                    out,
                )
            }
        }
        crate::format::model_config::Precision::F64 { runtime_input, .. } => {
            if let Some(benchmark_blob) = benchmark_override {
                let req = super::container::ContainerBuildRequest {
                    model_name: _model_name,
                    topology,
                    precision_str: "f64",
                    rmd1_bytes: rmd1,
                    benchmark_override: Some(benchmark_blob),
                    runtime_input: runtime_input.map(RuntimeInput::F64),
                    conv_spec: None,
                };
                build_container_from_rmd1_with_benchmark(
                    &req,
                    benchmark_blob,
                    metadata_scratch,
                    out,
                )
            } else {
                let req = super::container::ContainerBuildRequest {
                    model_name: _model_name,
                    topology,
                    precision_str: "f64",
                    rmd1_bytes: rmd1,
                    benchmark_override: None,
                    runtime_input: runtime_input.map(RuntimeInput::F64),
                    conv_spec: None,
                };
                build_container_from_rmd1(
                    &req,
                    metadata_scratch,
                    out,
                )
            }
        }
        crate::format::model_config::Precision::LmF32 { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::LmF64 { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::LmTrainF32 { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::LmTrainF64 { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::InferF32 { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::InferF64 { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::LmText { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
        crate::format::model_config::Precision::LmChat { .. } => {
            Err(crate::engine::RnnFlowError::InvalidTopology)
        }
    }
}
