use super::api::RnnFfiReadView;

pub(crate) const RNN_FFI_OK: i32 = 0;
pub(crate) const RNN_FFI_NULL_POINTER: i32 = 1;
pub(crate) const RNN_FFI_INVALID_ARGUMENT: i32 = 2;
pub(crate) const RNN_FFI_BAD_BYTES: i32 = 3;
pub(crate) const RNN_FFI_CAPACITY_TOO_SMALL: i32 = 4;
pub(crate) const RNN_FFI_INTERNAL: i32 = 8;

pub(crate) fn read_str<'a>(ptr: *const u8, len: usize) -> Result<&'a str, i32> {
    if ptr.is_null() {
        return Err(RNN_FFI_NULL_POINTER);
    }
    let bytes = unsafe { core::slice::from_raw_parts(ptr, len) };
    core::str::from_utf8(bytes).map_err(|_| RNN_FFI_INVALID_ARGUMENT)
}

pub(crate) fn nop_policy(m: &crate::format::model_config::TrainingMetrics) -> f64 {
    crate::engine::runtime::hardware::cpu_cap_80(m)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_f32(
    model_name_ptr: *const u8,
    model_name_len: usize,
    topology_ptr: *const usize,
    topology_len: usize,
    weights_ptr: *const f32,
    weights_len: usize,
    biases_ptr: *const f32,
    biases_len: usize,
    out_ptr: *mut u8,
    out_cap: usize,
    out_used: *mut usize,
) -> i32 {
    if model_name_ptr.is_null() || topology_ptr.is_null()
        || weights_ptr.is_null() || biases_ptr.is_null()
        || out_ptr.is_null() || out_used.is_null()
    {
        return RNN_FFI_NULL_POINTER;
    }
    let model_name = match read_str(model_name_ptr, model_name_len) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let topology = unsafe { core::slice::from_raw_parts(topology_ptr, topology_len) };
    let weights = unsafe { core::slice::from_raw_parts_mut(weights_ptr as *mut f32, weights_len) };
    let biases = unsafe { core::slice::from_raw_parts_mut(biases_ptr as *mut f32, biases_len) };
    let buf = unsafe { core::slice::from_raw_parts_mut(out_ptr, out_cap) };
    let req = crate::api::rnn_api::core_api::BuildF32Request {
        model_name,
        topology,
        seed: None,
        weights,
        biases,
        runtime_input: None,
        conv_spec: None,
        conv_in_shape: [0; 5],
        conv_layers: &[],
        benchmark_override: None,
        lm_params: &[],
        dataset_dirs: &[],
    };
    match crate::api::rnn_api::build::build_f32_with_request(req, buf) {
        Ok(used) => { unsafe { *out_used = used; } RNN_FFI_OK }
        Err(crate::api::rnn_api::core_api::RnnApiError::CapacityTooSmall) => RNN_FFI_CAPACITY_TOO_SMALL,
        Err(_) => RNN_FFI_INTERNAL,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_f64(
    model_name_ptr: *const u8,
    model_name_len: usize,
    topology_ptr: *const usize,
    topology_len: usize,
    weights_ptr: *const f64,
    weights_len: usize,
    biases_ptr: *const f64,
    biases_len: usize,
    out_ptr: *mut u8,
    out_cap: usize,
    out_used: *mut usize,
) -> i32 {
    if model_name_ptr.is_null() || topology_ptr.is_null()
        || weights_ptr.is_null() || biases_ptr.is_null()
        || out_ptr.is_null() || out_used.is_null()
    {
        return RNN_FFI_NULL_POINTER;
    }
    let model_name = match read_str(model_name_ptr, model_name_len) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let topology = unsafe { core::slice::from_raw_parts(topology_ptr, topology_len) };
    let weights = unsafe { core::slice::from_raw_parts_mut(weights_ptr as *mut f64, weights_len) };
    let biases = unsafe { core::slice::from_raw_parts_mut(biases_ptr as *mut f64, biases_len) };
    let buf = unsafe { core::slice::from_raw_parts_mut(out_ptr, out_cap) };
    let req = crate::api::rnn_api::core_api::BuildF64Request {
        model_name,
        topology,
        seed: None,
        weights,
        biases,
        runtime_input: None,
        benchmark_override: None,
        lm_params: &[],
        dataset_dirs: &[],
    };
    match crate::api::rnn_api::build::build_f64_with_request(req, buf) {
        Ok(used) => { unsafe { *out_used = used; } RNN_FFI_OK }
        Err(crate::api::rnn_api::core_api::RnnApiError::CapacityTooSmall) => RNN_FFI_CAPACITY_TOO_SMALL,
        Err(_) => RNN_FFI_INTERNAL,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn train(
    model_ptr: *mut u8,
    model_len: usize,
    topology_ptr: *const usize,
    topology_len: usize,
    input_ptr: *const f32,
    input_len: usize,
    target_ptr: *const f32,
    target_len: usize,
    learning_rate: f32,
    train_seconds: u64,
    start_iteration: u64,
    max_iterations: u64,
    out_avg_loss: *mut f64,
    out_iterations: *mut usize,
) -> i32 {
    if model_ptr.is_null() || topology_ptr.is_null()
        || input_ptr.is_null() || target_ptr.is_null()
        || out_avg_loss.is_null() || out_iterations.is_null()
    {
        return RNN_FFI_NULL_POINTER;
    }
    let bytes = unsafe { core::slice::from_raw_parts_mut(model_ptr, model_len) };
    let topology = unsafe { core::slice::from_raw_parts(topology_ptr, topology_len) };
    let input = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let target = unsafe { core::slice::from_raw_parts(target_ptr, target_len) };
    let sample = (input, target);
    let samples = core::slice::from_ref(&sample);
    let mut config = crate::format::model_config::ModelConfig {
        model_name: "",
        topology,
        precision: crate::format::model_config::Precision::F32 { weights: &[], biases: &[], runtime_input: None, conv_spec: None },
        benchmark_override: None,
        samples,
        learning_rate,
        gradient_clip: None,
        device_id: &[],
        trusted_publisher_pubkeys: &[],
        evolve_out: None,
        sample_filler: None,
        train_seconds,
        max_iterations: if max_iterations > 0 { Some(max_iterations as usize) } else { None },
        start_iteration: start_iteration as usize,
        elapsed_offset_ns: 0,
        progress_callback: None,
        progress_interval_ns: 0,
        policy: nop_policy,
        encrypted_data_len: None,
        evolve: None,
    };
    match crate::api::rnn_api::core_api::train(bytes, &mut config) {
        Ok((avg_loss, iterations)) => {
            unsafe { *out_avg_loss = avg_loss; *out_iterations = iterations; }
            RNN_FFI_OK
        }
        Err(crate::api::rnn_api::core_api::RnnApiError::BadBytes) => RNN_FFI_BAD_BYTES,
        Err(crate::api::rnn_api::core_api::RnnApiError::CapacityTooSmall) => RNN_FFI_CAPACITY_TOO_SMALL,
        Err(_) => RNN_FFI_INTERNAL,
    }
}

pub(crate) fn run(
    model_ptr: *mut u8,
    model_len: usize,
    topology_ptr: *const usize,
    topology_len: usize,
    input_ptr: *const f32,
    input_len: usize,
    out_score: *mut f64,
) -> i32 {
    if model_ptr.is_null() || topology_ptr.is_null()
        || input_ptr.is_null() || out_score.is_null()
    {
        return RNN_FFI_NULL_POINTER;
    }
    let bytes = unsafe { core::slice::from_raw_parts_mut(model_ptr, model_len) };
    let topology = unsafe { core::slice::from_raw_parts(topology_ptr, topology_len) };
    let input = unsafe { core::slice::from_raw_parts(input_ptr, input_len) };
    let mut config = crate::format::model_config::ModelConfig {
        model_name: "",
        topology,
        precision: crate::format::model_config::Precision::F32 { weights: &[], biases: &[], runtime_input: Some(input), conv_spec: None },
        benchmark_override: None,
        samples: &[],
        learning_rate: 0.0,
        gradient_clip: None,
        device_id: &[],
        trusted_publisher_pubkeys: &[],
        evolve_out: None,
        sample_filler: None,
        train_seconds: 0,
        max_iterations: Some(1),
        start_iteration: 0,
        elapsed_offset_ns: 0,
        progress_callback: None,
        progress_interval_ns: 0,
        policy: nop_policy,
        encrypted_data_len: None,
        evolve: None,
    };
    match crate::api::rnn_api::run::run(bytes, &mut config) {
        Ok((score, _)) => {
            unsafe { *out_score = score; }
            RNN_FFI_OK
        }
        Err(crate::api::rnn_api::core_api::RnnApiError::BadBytes) => RNN_FFI_BAD_BYTES,
        Err(_) => RNN_FFI_INTERNAL,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn read_rnn(
    model_ptr: *mut u8,
    model_len: usize,
    device_id_ptr: *const u8,
    device_id_len: usize,
    out_topology_ptr: *mut usize,
    out_topology_cap: usize,
    out_view: *mut RnnFfiReadView,
) -> i32 {
    if model_ptr.is_null() || out_view.is_null() {
        return RNN_FFI_NULL_POINTER;
    }
    let bytes = unsafe { core::slice::from_raw_parts_mut(model_ptr, model_len) };
    let device_id = if device_id_ptr.is_null() {
        None
    } else {
        Some(unsafe { core::slice::from_raw_parts(device_id_ptr, device_id_len) })
    };
    let readable = match crate::api::rnn_api::read::read_rnn(bytes, device_id) {
        Ok(r) => r,
        Err(crate::api::rnn_api::core_api::RnnApiError::BadBytes) => return RNN_FFI_BAD_BYTES,
        Err(crate::api::rnn_api::core_api::RnnApiError::CapacityTooSmall) => return RNN_FFI_CAPACITY_TOO_SMALL,
        Err(_) => return RNN_FFI_INTERNAL,
    };
    if readable.topo_len > out_topology_cap {
        return RNN_FFI_CAPACITY_TOO_SMALL;
    }
    if readable.topo_len > 0 {
        if out_topology_ptr.is_null() {
            return RNN_FFI_NULL_POINTER;
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                readable.topology.as_ptr(),
                out_topology_ptr,
                readable.topo_len,
            );
        }
    }
    let (name_ptr, name_len) = opt_str(readable.model_name);
    let (precision_ptr, precision_len) = opt_str(readable.model_precision);
    let (layer_meta_ptr, layer_meta_len) = opt_bytes(readable.layer_meta);
    let (weights_ptr, weights_len) = opt_bytes(readable.weights);
    let (biases_ptr, biases_len) = opt_bytes(readable.biases);
    let (benchmark_ptr, benchmark_len) = opt_bytes(readable.benchmark);
    let (training_log_ptr, training_log_len) = opt_bytes(readable.training_log);
    let view = RnnFfiReadView {
        model_name_ptr: name_ptr,
        model_name_len: name_len,
        model_precision_ptr: precision_ptr,
        model_precision_len: precision_len,
        layer_meta_ptr,
        layer_meta_len,
        weights_ptr,
        weights_len,
        biases_ptr,
        biases_len,
        benchmark_ptr,
        benchmark_len,
        training_log_ptr,
        training_log_len,
        topology_len: readable.topo_len,
        elapsed_ms: readable.elapsed_ms,
        iterations: readable.iterations,
        train_samples: readable.train_samples,
        avg_loss: readable.avg_loss,
        last_loss: readable.last_loss,
        output_bytes: readable.output_bytes,
        total_params: readable.total_params,
        layer_count: readable.layer_count,
        input_dim: readable.input_dim,
        output_dim: readable.output_dim,
        benchmark_flags: readable.benchmark_flags,
        weights_bytes: readable.weights_bytes,
        biases_bytes: readable.biases_bytes,
        min_loss: readable.min_loss,
        max_loss: readable.max_loss,
        loss_stddev: readable.loss_stddev,
        iterations_per_sec: readable.iterations_per_sec,
        samples_per_sec: readable.samples_per_sec,
        eval_loss: readable.eval_loss,
        eval_accuracy: readable.eval_accuracy,
        eval_f1: readable.eval_f1,
        eval_mae: readable.eval_mae,
        eval_samples: readable.eval_samples,
        eval_dataset_hash: readable.eval_dataset_hash,
        logical_cores: readable.logical_cores,
        avg_frequency_mhz: readable.avg_frequency_mhz,
        max_frequency_mhz: readable.max_frequency_mhz,
        max_workers: readable.max_workers,
        target_cpu_utilization: readable.target_cpu_utilization,
    };
    unsafe { *out_view = view; }
    RNN_FFI_OK
}

fn opt_str(value: Option<&str>) -> (*const u8, usize) {
    match value {
        Some(s) => (s.as_ptr(), s.len()),
        None => (core::ptr::null(), 0),
    }
}

fn opt_bytes(value: Option<&[u8]>) -> (*const u8, usize) {
    match value {
        Some(b) => (b.as_ptr(), b.len()),
        None => (core::ptr::null(), 0),
    }
}
