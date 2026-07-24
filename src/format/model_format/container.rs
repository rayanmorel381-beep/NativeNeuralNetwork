use crate::observability::benchmark::{
    decode_benchmark_blob, encode_benchmark_blob, encoded_size_benchmark_blob, BenchmarkEncodeError,
    BenchmarkMetrics,
};
use crate::engine::rnn_flow::{
    blob_payload_by_name, map_protocol_error, rnn_dtype, RnnFlowError,
};
use crate::format::model_config::ingest;
use crate::format::model_format::{
    encode_blob_payloads_with_header_metadata,
    neuron_positions_blob_into, neuron_positions_blob_size, neuron_positions_rows, parse_payload,
    total_neurons_from_layer_meta, BlobDesc, NEURON_POSITION_COLS,
};
use crate::format::model_format::{
    BLOB_BIASES, BLOB_CONV_SPEC, BLOB_LAYER_META, BLOB_NEURON_POSITIONS, BLOB_RUNTIME_INPUT,
    BLOB_WEIGHTS,
};
use crate::format::model_format::{LMLP_GRAPH_BLOB_DATA, LMLP_TENSORS_BLOB_DATA};
use crate::format::model_format::lmlp::{
    encode_lmlp_graph_into, LmlpEdgeView, LmlpNodeView, LMLP_GRAPH_VERSION,
};
use crate::format::rnn_format::lookup::find_blob_index;
use crate::format::rnn_format::parser::parse_rnn_from_bytes;
use crate::base::scratch::Scratch;
use crate::base::tensor::TensorView;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReadableRnnFormat<'a> {
    pub scan: crate::format::rnn_format::scanner::ScanReport,
    pub model_name: Option<&'a str>,
    pub model_precision: Option<&'a str>,
    pub layer_meta: Option<&'a [u8]>,
    pub weights: Option<&'a [u8]>,
    pub biases: Option<&'a [u8]>,
    pub benchmark: Option<&'a [u8]>,
    pub training_log: Option<&'a [u8]>,
    pub topology: [usize; 128],
    pub topo_len: usize,
    pub elapsed_ms: u64,
    pub iterations: u64,
    pub train_samples: u64,
    pub avg_loss: f32,
    pub last_loss: f32,
    pub output_bytes: u64,
    pub total_params: u64,
    pub layer_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub benchmark_flags: u64,
    pub weights_bytes: u64,
    pub biases_bytes: u64,
    pub min_loss: f32,
    pub max_loss: f32,
    pub loss_stddev: f32,
    pub iterations_per_sec: f32,
    pub samples_per_sec: f32,
    pub eval_loss: f32,
    pub eval_accuracy: f32,
    pub eval_f1: f32,
    pub eval_mae: f32,
    pub eval_samples: u64,
    pub eval_dataset_hash: u64,
    pub logical_cores: u32,
    pub avg_frequency_mhz: u32,
    pub max_frequency_mhz: u32,
    pub max_workers: u32,
    pub target_cpu_utilization: f32,
}

#[derive(Clone, Copy)]
pub struct BlobWrite<'a> {
    pub name: &'a str,
    pub payload: &'a [u8],
}

#[derive(Clone, Copy)]
pub enum RuntimeInput<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

pub fn validate_benchmark_flags(topology: &[usize], precision: &str) -> Result<u64, RnnFlowError> {
    let request = crate::format::model_config::rnn::ValidateRequest::BuildBenchmark {
        topology,
        precision,
    };
    match crate::format::model_config::rnn::validate(request) {
        Ok(crate::format::model_config::rnn::ValidateResult::BuildBenchmarkFlags(flags)) => Ok(flags),
        Ok(_) => Err(RnnFlowError::InvalidTopology),
        Err(_) => Err(RnnFlowError::InvalidTopology),
    }
}

pub fn build_default_benchmark_blob(
    model_name: &str,
    precision: &str,
    topology: &[usize],
    output_bytes: usize,
    benchmark_flags: u64,
    train_samples: u64,
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let (weights, biases) = crate::base::initializers::expected_parameter_counts(topology)
        .ok_or(RnnFlowError::InvalidTopology)?;
    let layer_count = topology.len().saturating_sub(1) as u32;
    let input_dim = topology.first().copied().unwrap_or(0) as u32;
    let output_dim = topology.last().copied().unwrap_or(0) as u32;
    let element_size = if precision.eq_ignore_ascii_case("f64") {
        core::mem::size_of::<f64>() as u64
    } else {
        core::mem::size_of::<f32>() as u64
    };
    let metrics = BenchmarkMetrics {
        model_name,
        precision,
        elapsed_ms: 0,
        iterations: 0,
        train_samples,
        avg_loss: 0.0,
        last_loss: 0.0,
        output_bytes,
        total_params: weights.saturating_add(biases) as u64,
        layer_count,
        input_dim,
        output_dim,
        benchmark_flags,
        weights_bytes: (weights as u64).saturating_mul(element_size),
        biases_bytes: (biases as u64).saturating_mul(element_size),
        min_loss: 0.0,
        max_loss: 0.0,
        loss_stddev: 0.0,
        iterations_per_sec: 0.0,
        samples_per_sec: 0.0,
        eval_loss: 0.0,
        eval_accuracy: 0.0,
        eval_f1: 0.0,
        eval_mae: 0.0,
        eval_samples: 0,
        eval_dataset_hash: 0,
        logical_cores: 0,
        avg_frequency_mhz: 0,
        max_frequency_mhz: 0,
        max_workers: 0,
        target_cpu_utilization: 0.0,
    };
    let needed = encoded_size_benchmark_blob(&metrics).ok_or(RnnFlowError::InvalidTopology)?;
    if out.len() < needed {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    encode_benchmark_blob(&metrics, out).map_err(map_benchmark_encode_error)
}

fn map_benchmark_encode_error(err: BenchmarkEncodeError) -> RnnFlowError {
    match err {
        BenchmarkEncodeError::BufferTooSmall => RnnFlowError::CapacityTooSmall,
        BenchmarkEncodeError::InvalidFormat => RnnFlowError::BadBytes,
    }
}

pub fn read_rnn_format<'a>(
    bytes: &'a [u8],
    device_id: Option<&[u8]>,
) -> Result<ReadableRnnFormat<'a>, RnnFlowError> {
    let dtype = rnn_dtype(bytes)?;
    if !matches!(dtype, 0 | 1) {
        return Err(RnnFlowError::BadBytes);
    }
    if crate::format::model_format::parse_header(bytes).is_some()
        && !crate::format::model_format::payload_bounds::has_full_payload(bytes)
    {
        return Err(RnnFlowError::BadBytes);
    }
    let scan = crate::format::rnn_format::validate(bytes, device_id).map_err(|_| RnnFlowError::BadBytes)?;
    let mut readable = ReadableRnnFormat {
        scan,
        model_name: None,
        model_precision: None,
        layer_meta: None,
        weights: None,
        biases: None,
        benchmark: None,
        training_log: None,
        topology: [0usize; 128],
        topo_len: 0,
        elapsed_ms: 0,
        iterations: 0,
        train_samples: 0,
        avg_loss: 0.0,
        last_loss: 0.0,
        output_bytes: 0,
        total_params: 0,
        layer_count: 0,
        input_dim: 0,
        output_dim: 0,
        benchmark_flags: 0,
        weights_bytes: 0,
        biases_bytes: 0,
        min_loss: 0.0,
        max_loss: 0.0,
        loss_stddev: 0.0,
        iterations_per_sec: 0.0,
        samples_per_sec: 0.0,
        eval_loss: 0.0,
        eval_accuracy: 0.0,
        eval_f1: 0.0,
        eval_mae: 0.0,
        eval_samples: 0,
        eval_dataset_hash: 0,
        logical_cores: 0,
        avg_frequency_mhz: 0,
        max_frequency_mhz: 0,
        max_workers: 0,
        target_cpu_utilization: 0.0,
    };
    if !scan.encrypted {
        let mut scratch_buf = [0u8; 16384];
        let mut scratch = Scratch::new(&mut scratch_buf);
        let handle =
            parse_rnn_from_bytes(bytes, &mut scratch).map_err(|_| RnnFlowError::BadBytes)?;
        readable.layer_meta = blob_payload_by_name(bytes, &handle, BLOB_LAYER_META);
        readable.weights = blob_payload_by_name(bytes, &handle, BLOB_WEIGHTS);
        readable.biases = blob_payload_by_name(bytes, &handle, BLOB_BIASES);
        readable.benchmark =
            crate::format::model_format::header_tlv_payload(bytes, ingest::TLV_HEADER_BENCHMARK).ok();
        readable.model_name =
            crate::format::model_format::header_tlv_payload(bytes, ingest::TLV_HEADER_MODEL_NAME)
                .ok()
                .and_then(|v| core::str::from_utf8(v).ok());
        readable.model_precision = match rnn_dtype(bytes) {
            Ok(0) => Some("f32"),
            Ok(1) => Some("f64"),
            _ => None,
        };
        readable.training_log =
            blob_payload_by_name(bytes, &handle, crate::format::model_format::BLOB_TRAINING_LOG);
        if let Some(meta) = readable.layer_meta {
            let entry_size = 20usize;
            if meta.len() >= entry_size && meta.len() % entry_size == 0 {
                let layer_count = meta.len() / entry_size;
                let mut tl = 0usize;
                for i in 0..layer_count {
                    let off = i * entry_size;
                    let inp = u32::from_le_bytes([meta[off], meta[off+1], meta[off+2], meta[off+3]]) as usize;
                    let out = u32::from_le_bytes([meta[off+4], meta[off+5], meta[off+6], meta[off+7]]) as usize;
                    if i == 0 && tl < 128 { readable.topology[tl] = inp; tl += 1; }
                    if tl < 128 { readable.topology[tl] = out; tl += 1; }
                }
                readable.topo_len = tl;
            }
        }

        let elem_size = if dtype == 1 { 8usize } else { 4usize };
        if let Some(w) = readable.weights {
            readable.weights_bytes = w.len() as u64;
        }
        if let Some(b) = readable.biases {
            readable.biases_bytes = b.len() as u64;
        }
        let weights_params = readable.weights.map_or(0usize, |w| w.len() / elem_size) as u64;
        let biases_params = readable.biases.map_or(0usize, |b| b.len() / elem_size) as u64;
        readable.total_params = weights_params + biases_params;
        if readable.topo_len > 0 {
            readable.layer_count = (readable.topo_len - 1) as u32;
            readable.input_dim = readable.topology[0] as u32;
            readable.output_dim = readable.topology[readable.topo_len - 1] as u32;
        }
        if let Ok(view) = crate::observability::benchmark::get_bmk(bytes) {
            readable.elapsed_ms = view.elapsed_ms;
            readable.iterations = view.iterations;
            readable.train_samples = view.train_samples;
            readable.avg_loss = view.avg_loss;
            readable.last_loss = view.last_loss;
            readable.output_bytes = view.output_bytes as u64;
            readable.benchmark_flags = view.benchmark_flags;
            readable.min_loss = view.min_loss;
            readable.max_loss = view.max_loss;
            readable.loss_stddev = view.loss_stddev;
            readable.iterations_per_sec = view.iterations_per_sec;
            readable.samples_per_sec = view.samples_per_sec;
            readable.eval_loss = view.eval_loss;
            readable.eval_accuracy = view.eval_accuracy;
            readable.eval_f1 = view.eval_f1;
            readable.eval_mae = view.eval_mae;
            readable.eval_samples = view.eval_samples;
            readable.eval_dataset_hash = view.eval_dataset_hash;
            readable.logical_cores = view.logical_cores;
            readable.avg_frequency_mhz = view.avg_frequency_mhz;
            readable.max_frequency_mhz = view.max_frequency_mhz;
            readable.max_workers = view.max_workers;
            readable.target_cpu_utilization = view.target_cpu_utilization;
        }
    }
    Ok(readable)
}


pub fn fill_blob_payloads(bytes: &mut [u8], writes: &[BlobWrite<'_>]) -> Result<(), RnnFlowError> {
    for write in writes {
        let (start, end, len) = {
            let mut scratch_buf = [0u8; 16384];
            let mut scratch = Scratch::new(&mut scratch_buf);
            let handle =
                parse_rnn_from_bytes(bytes, &mut scratch).map_err(|_| RnnFlowError::BadBytes)?;
            let idx = find_blob_index(&handle, write.name).ok_or(RnnFlowError::Model)?;
            let desc = handle.blobs.get(idx).ok_or(RnnFlowError::Model)?;
            let start = usize::try_from(desc.offset).map_err(|_| RnnFlowError::BadBytes)?;
            let len = usize::try_from(desc.length).map_err(|_| RnnFlowError::BadBytes)?;
            let end = start.checked_add(len).ok_or(RnnFlowError::BadBytes)?;
            (start, end, len)
        };
        if end > bytes.len() || len != write.payload.len() {
            return Err(RnnFlowError::BadBytes);
        }
        bytes[start..end].copy_from_slice(write.payload);
    }
    Ok(())
}

pub(crate) fn assemble_rnn_container(
    records: &[BlobDesc<'_>],
    model_name_header: Option<&str>,
    benchmark_header: &[u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let used = encode_blob_payloads_with_header_metadata(
        records,
        model_name_header,
        Some(benchmark_header),
        out_bytes,
    )
    .map_err(map_protocol_error)?;
    Ok(used)
}

pub fn build_container_from_rmd1(
    req: &ContainerBuildRequest<'_>,
    metadata_scratch: &mut [u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    build_container_from_rmd1_with_runtime_input(req, metadata_scratch, out_bytes)
}

pub fn build_container_from_rmd1_with_benchmark(
    req: &ContainerBuildRequest<'_>,
    benchmark_blob: &[u8],
    metadata_scratch: &mut [u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let req = ContainerBuildRequest {
        benchmark_override: Some(benchmark_blob),
        ..*req
    };
    build_container_from_rmd1_with_runtime_input(&req, metadata_scratch, out_bytes)
}

pub fn build_container_from_rmd1_with_runtime_input(
    req: &ContainerBuildRequest<'_>,
    metadata_scratch: &mut [u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    build_container_from_rmd1_internal(req, metadata_scratch, out_bytes)
}

pub fn build_container_from_rmd1_with_conv_spec(
    req: &ContainerBuildRequest<'_>,
    metadata_scratch: &mut [u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    build_container_from_rmd1_internal_named(req, metadata_scratch, out_bytes)
}

pub(crate) fn build_container_from_rmd1_internal(
    req: &ContainerBuildRequest<'_>,
    metadata_scratch: &mut [u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    build_container_from_rmd1_internal_named(req, metadata_scratch, out_bytes)
}

#[derive(Clone, Copy)]
pub struct ContainerBuildRequest<'a> {
    pub model_name: &'a str,
    pub topology: &'a [usize],
    pub precision_str: &'a str,
    pub rmd1_bytes: &'a [u8],
    pub benchmark_override: Option<&'a [u8]>,
    pub runtime_input: Option<RuntimeInput<'a>>,
    pub conv_spec: Option<&'a [u8]>,
}

fn build_dense_lmlp_tensors(
    topology: &[usize],
    dtype: u8,
    weights: &[u8],
    biases: &[u8],
    out: &mut [u8],
) -> Option<usize> {
    let layer_count = topology.len().checked_sub(1)?;
    if layer_count == 0 {
        return None;
    }
    let esz: usize = if dtype == 1 { 8 } else { 4 };
    let mut cursor = 0usize;
    let mut w_off = 0usize;
    let mut b_off = 0usize;
    for i in 0..layer_count {
        let in_i = topology[i];
        let out_i = topology[i + 1];
        let w_bytes = in_i.checked_mul(out_i)?.checked_mul(esz)?;
        let b_bytes = out_i.checked_mul(esz)?;
        let w_end = w_off.checked_add(w_bytes)?;
        let b_end = b_off.checked_add(b_bytes)?;
        if w_end > weights.len() || b_end > biases.len() {
            return None;
        }
        let c_w_end = cursor.checked_add(w_bytes)?;
        if c_w_end > out.len() {
            return None;
        }
        out[cursor..c_w_end].copy_from_slice(&weights[w_off..w_end]);
        cursor = c_w_end;
        let c_b_end = cursor.checked_add(b_bytes)?;
        if c_b_end > out.len() {
            return None;
        }
        out[cursor..c_b_end].copy_from_slice(&biases[b_off..b_end]);
        cursor = c_b_end;
        w_off = w_end;
        b_off = b_end;
    }
    Some(cursor)
}

fn build_dense_lmlp_graph(topology: &[usize], dtype: u8, out: &mut [u8]) -> Option<usize> {
    let layer_count = topology.len().checked_sub(1)?;
    if layer_count == 0 || layer_count > 64 {
        return None;
    }
    let dtype_tag = if dtype == 1 { 1u8 } else { 0u8 };
    let zero_node = LmlpNodeView {
        kind_transform: 0,
        memory_flags: 0,
        norm_flags: 0,
        activation: 0,
        in_dim: 0,
        out_dim: 0,
        theta_offset: 0,
        theta_len: 0,
    };
    let zero_edge = LmlpEdgeView {
        from: 0,
        to: 0,
        default_mode: 0,
        w_offset: 0,
        w_len: 0,
        delay_hint: 0,
        phase: 0.0,
        decay: 1.0,
    };
    let mut nodes = [zero_node; 64];
    let mut edges = [zero_edge; 64];
    let mut theta = 0u32;
    for i in 0..layer_count {
        let in_i = topology[i];
        let out_i = topology[i + 1];
        let node_theta_len = in_i.checked_mul(out_i)?.checked_add(out_i)?;
        let tl = u32::try_from(node_theta_len).ok()?;
        nodes[i] = LmlpNodeView {
            kind_transform: 4,
            memory_flags: 0,
            norm_flags: 0,
            activation: 0,
            in_dim: u32::try_from(in_i).ok()?,
            out_dim: u32::try_from(out_i).ok()?,
            theta_offset: theta,
            theta_len: tl,
        };
        theta = theta.checked_add(tl)?;
    }
    let edge_count = layer_count - 1;
    for i in 0..edge_count {
        edges[i] = LmlpEdgeView {
            from: u32::try_from(i).ok()?,
            to: u32::try_from(i + 1).ok()?,
            default_mode: 0,
            w_offset: 0,
            w_len: 0,
            delay_hint: 0,
            phase: 0.0,
            decay: 1.0,
        };
    }
    let in_ids = [0u32];
    let out_ids = [u32::try_from(layer_count - 1).ok()?];
    encode_lmlp_graph_into(
        LMLP_GRAPH_VERSION,
        dtype_tag,
        &nodes[..layer_count],
        &edges[..edge_count],
        &in_ids,
        &out_ids,
        0,
        out,
    )
}

pub(crate) fn build_container_from_rmd1_internal_named(
    req: &ContainerBuildRequest<'_>,
    metadata_scratch: &mut [u8],
    out_bytes: &mut [u8],
) -> Result<usize, RnnFlowError> {
    let preserve_existing_benchmark = matches!(req.benchmark_override, Some(blob) if !blob.is_empty());
    let benchmark_used = if let Some(benchmark_blob) = req.benchmark_override {
        if benchmark_blob.is_empty() {
            let benchmark_flags = validate_benchmark_flags(req.topology, req.precision_str)?;
            build_default_benchmark_blob(
                req.model_name,
                req.precision_str,
                req.topology,
                0,
                benchmark_flags,
                0,
                metadata_scratch,
            )?
        } else {
            decode_benchmark_blob(benchmark_blob).map_err(|_| RnnFlowError::BadBytes)?;
            if benchmark_blob.len() > metadata_scratch.len() {
                return Err(RnnFlowError::CapacityTooSmall);
            }
            metadata_scratch[..benchmark_blob.len()].copy_from_slice(benchmark_blob);
            benchmark_blob.len()
        }
    } else {
        let benchmark_flags = validate_benchmark_flags(req.topology, req.precision_str)?;
        build_default_benchmark_blob(
            req.model_name,
            req.precision_str,
            req.topology,
            req.rmd1_bytes.len(),
            benchmark_flags,
            0,
            metadata_scratch,
        )?
    };

    let dense = parse_payload(req.rmd1_bytes).map_err(map_protocol_error)?;
    let total_neurons =
        total_neurons_from_layer_meta(dense.layer_meta).map_err(map_protocol_error)?;
    let neuron_rows = neuron_positions_rows(dense.layer_meta).map_err(map_protocol_error)?;
    if neuron_rows != total_neurons + 1 {
        return Err(RnnFlowError::Model);
    }
    let neuron_positions_len =
        neuron_positions_blob_size(dense.dtype, dense.layer_meta).map_err(map_protocol_error)?;

    let (benchmark_area, remaining_meta) = metadata_scratch.split_at_mut(benchmark_used);
    if remaining_meta.len() < neuron_positions_len {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    let used_neuron_positions =
        neuron_positions_blob_into(dense.dtype, dense.layer_meta, dense.biases, remaining_meta)
            .map_err(map_protocol_error)?;

    let (neuron_positions_area, runtime_area) = remaining_meta.split_at_mut(used_neuron_positions);
    let input_dim = u32::from_le_bytes([
        dense.layer_meta[0],
        dense.layer_meta[1],
        dense.layer_meta[2],
        dense.layer_meta[3],
    ]) as usize;

    let runtime_input_len_opt = match req.runtime_input {
        Some(RuntimeInput::F32(values)) => Some(encode_runtime_input_from_f32(
            dense.dtype, input_dim, values, runtime_area,
        )?),
        Some(RuntimeInput::F64(values)) => Some(encode_runtime_input_from_f64(
            dense.dtype, input_dim, values, runtime_area,
        )?),
        None => None,
    };

    let benchmark_blob = &benchmark_area[..benchmark_used];
    let neuron_positions_blob = &neuron_positions_area[..used_neuron_positions];

    let runtime_used = runtime_input_len_opt.unwrap_or(0);
    let (rt_slot, tensors_area) = runtime_area.split_at_mut(runtime_used);
    let runtime_input_blob: Option<&[u8]> = if runtime_used > 0 {
        Some(&rt_slot[..runtime_used])
    } else {
        None
    };

    let lmlp_tensors_len =
        build_dense_lmlp_tensors(req.topology, dense.dtype, dense.weights, dense.biases, tensors_area);
    let lmlp_tensors_blob: Option<&[u8]> = match lmlp_tensors_len {
        Some(n) if n > 0 => Some(&tensors_area[..n]),
        _ => None,
    };

    let mut records_arr: [BlobDesc; 8] = [
        BlobDesc {
            name: BLOB_NEURON_POSITIONS,
            dtype: dense.dtype,
            dims: [(total_neurons + 1) as u32, NEURON_POSITION_COLS as u32],
            ndim: 2,
            payload: neuron_positions_blob,
        },
        BlobDesc {
            name: BLOB_LAYER_META,
            dtype: dense.dtype,
            dims: [dense.layer_count as u32, 5],
            ndim: 2,
            payload: dense.layer_meta,
        },
        BlobDesc {
            name: BLOB_WEIGHTS,
            dtype: dense.dtype,
            dims: [dense.weights_len as u32, 0],
            ndim: 1,
            payload: dense.weights,
        },
        BlobDesc {
            name: BLOB_BIASES,
            dtype: dense.dtype,
            dims: [dense.biases_len as u32, 0],
            ndim: 1,
            payload: dense.biases,
        },
        BlobDesc {
            name: BLOB_RUNTIME_INPUT,
            dtype: dense.dtype,
            dims: [0u32, 0],
            ndim: 0,
            payload: &[],
        },
        BlobDesc {
            name: BLOB_CONV_SPEC,
            dtype: 0,
            dims: [0u32, 0],
            ndim: 0,
            payload: &[],
        },
        BlobDesc {
            name: BLOB_CONV_SPEC,
            dtype: 0,
            dims: [0u32, 0],
            ndim: 0,
            payload: &[],
        },
        BlobDesc {
            name: BLOB_CONV_SPEC,
            dtype: 0,
            dims: [0u32, 0],
            ndim: 0,
            payload: &[],
        },
    ];

    let mut records_count = 4usize;
    if let Some(blob) = runtime_input_blob {
        records_arr[4] = BlobDesc {
            name: BLOB_RUNTIME_INPUT,
            dtype: dense.dtype,
            dims: [input_dim as u32, 0],
            ndim: 1,
            payload: blob,
        };
        records_count = 5;
    }
    if let Some(conv_blob) = req.conv_spec {
        if conv_blob.is_empty() {
            return Err(RnnFlowError::BadBytes);
        }
        if dense.dtype != 0 {
            return Err(RnnFlowError::BadBytes);
        }
        crate::graph::conv::conv_net::conv_net_validate(conv_blob).map_err(|_| RnnFlowError::BadBytes)?;
        let conv_out =
            crate::graph::conv::conv_net::conv_net_output_len(conv_blob).map_err(|_| RnnFlowError::BadBytes)?;
        if conv_out != input_dim {
            return Err(RnnFlowError::BadBytes);
        }
        let conv_len = u32::try_from(conv_blob.len()).map_err(|_| RnnFlowError::BadBytes)?;
        records_arr[records_count] = BlobDesc {
            name: BLOB_CONV_SPEC,
            dtype: 0,
            dims: [conv_len, 0],
            ndim: 1,
            payload: conv_blob,
        };
        records_count += 1;
    }

    if let Some(blob) = lmlp_tensors_blob {
        if records_count < records_arr.len() {
            records_arr[records_count] = BlobDesc {
                name: LMLP_TENSORS_BLOB_DATA,
                dtype: dense.dtype,
                dims: [u32::try_from(blob.len()).unwrap_or(u32::MAX), 0],
                ndim: 1,
                payload: blob,
            };
            records_count += 1;
        }
    }

    let mut lmlp_graph_buf = [0u8; 4096];
    if let Some(glen) = build_dense_lmlp_graph(req.topology, dense.dtype, &mut lmlp_graph_buf) {
        if records_count < records_arr.len() {
            records_arr[records_count] = BlobDesc {
                name: LMLP_GRAPH_BLOB_DATA,
                dtype: dense.dtype,
                dims: [glen as u32, 0],
                ndim: 1,
                payload: &lmlp_graph_buf[..glen],
            };
            records_count += 1;
        }
    }

    let records_slice = &records_arr[..records_count];
    crate::format::model_config::rnn::validate_payload_records(records_slice)
        .map_err(|_| RnnFlowError::Model)?;

    let name_opt = if req.model_name.is_empty() { None } else { Some(req.model_name) };
    let used = assemble_rnn_container(records_slice, name_opt, benchmark_blob, out_bytes)?;

    if let Some(blob) = runtime_input_blob {
        fill_blob_payloads(out_bytes, &[BlobWrite { name: BLOB_RUNTIME_INPUT, payload: blob }])
            .map_err(|_| RnnFlowError::BadBytes)?;
    }
    if let Some(conv_blob) = req.conv_spec {
        fill_blob_payloads(out_bytes, &[BlobWrite { name: BLOB_CONV_SPEC, payload: conv_blob }])
            .map_err(|_| RnnFlowError::BadBytes)?;
    }

    if !preserve_existing_benchmark {
        let benchmark_flags = validate_benchmark_flags(req.topology, req.precision_str)?;
        let final_benchmark_used = build_default_benchmark_blob(
            req.model_name,
            req.precision_str,
            req.topology,
            used,
            benchmark_flags,
            0,
            metadata_scratch,
        )?;
        let (benchmark_start, benchmark_end) =
            crate::format::model_config::rnn::parse_header_tlv_range(&out_bytes[..used], ingest::TLV_HEADER_BENCHMARK)
                .map_err(|_| RnnFlowError::BadBytes)?;
        let slot_size = benchmark_end - benchmark_start;
        if slot_size == final_benchmark_used {
            out_bytes[benchmark_start..benchmark_end]
                .copy_from_slice(&metadata_scratch[..final_benchmark_used]);
        } else if slot_size <= crate::format::model_format::COMPACT_BENCHMARK_LEN
            && final_benchmark_used >= 168 {
            let compact = crate::format::model_format::compact_benchmark_for_header(
                &metadata_scratch[..final_benchmark_used],
            );
            out_bytes[benchmark_start..benchmark_end]
                .copy_from_slice(&compact[..slot_size]);
        } else if slot_size < final_benchmark_used {
            out_bytes[benchmark_start..benchmark_end]
                .copy_from_slice(&metadata_scratch[..slot_size]);
        } else {
            return Err(RnnFlowError::BadBytes);
        }
    }

    crate::observability::benchmark::get_bmk(&out_bytes[..used]).map_err(|_| RnnFlowError::BadBytes)?;
    Ok(used)
}

fn encode_runtime_input_from_f32(
    dtype: u8,
    input_dim: usize,
    values: &[f32],
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    if dtype != 0 || values.len() != input_dim {
        return Err(RnnFlowError::BadBytes);
    }
    let needed = input_dim
        .checked_mul(4)
        .ok_or(RnnFlowError::CapacityTooSmall)?;
    let align = core::mem::align_of::<f32>();
    let addr = out.as_ptr() as usize;
    let pad = (align - (addr % align)) % align;
    if pad
        .checked_add(needed)
        .ok_or(RnnFlowError::CapacityTooSmall)?
        > out.len()
    {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    let tensor_data: &mut [f32] = unsafe {
        core::slice::from_raw_parts_mut(out.as_mut_ptr().add(pad) as *mut f32, input_dim)
    };
    let mut tensor = TensorView {
        data: tensor_data,
        shape: [1, 1, 1, 1, input_dim],
    };
    if !tensor.is_valid_layout() {
        return Err(RnnFlowError::BadBytes);
    }
    if !(tensor.as_ptr() as usize).is_multiple_of(align) {
        return Err(RnnFlowError::BadBytes);
    }
    crate::base::tensor::tensor_fill(&mut tensor, 0.0);
    for (i, value) in values.iter().take(input_dim).enumerate() {
        if let Some(cell) = tensor.get_mut(0, 0, 0, 0, i) {
            *cell = *value;
        } else {
            return Err(RnnFlowError::BadBytes);
        }
    }
    for (i, value) in values.iter().take(input_dim).enumerate() {
        match tensor.get(0, 0, 0, 0, i) {
            Some(stored) if stored == *value => {}
            _ => return Err(RnnFlowError::BadBytes),
        }
    }
    if pad != 0 {
        out.copy_within(pad..pad + needed, 0);
    }
    Ok(needed)
}

fn encode_runtime_input_from_f64(
    dtype: u8,
    input_dim: usize,
    values: &[f64],
    out: &mut [u8],
) -> Result<usize, RnnFlowError> {
    if dtype != 1 || values.len() != input_dim {
        return Err(RnnFlowError::BadBytes);
    }
    let needed = input_dim
        .checked_mul(8)
        .ok_or(RnnFlowError::CapacityTooSmall)?;
    if needed > out.len() {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    for i in 0..input_dim {
        out[i * 8..i * 8 + 8].copy_from_slice(&values[i].to_le_bytes());
    }
    Ok(needed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_blob_payloads_rejects_invalid_layout() {
        let mut bytes = [0u8; 8];
        let writes = [BlobWrite { name: "missing", payload: &[] }];
        assert!(fill_blob_payloads(&mut bytes, &writes).is_err());
    }

    #[test]
    fn build_container_from_rmd1_rejects_invalid_inputs() {
        let mut scratch = [0u8; 512];
        let mut out = [0u8; 256];
        let req = ContainerBuildRequest {
            model_name: "",
            topology: &[2],
            precision_str: "f32",
            rmd1_bytes: &[],
            benchmark_override: None,
            runtime_input: None,
            conv_spec: None,
        };
        let result = build_container_from_rmd1(&req, &mut scratch, &mut out);
        assert!(result.is_err());
    }
}
