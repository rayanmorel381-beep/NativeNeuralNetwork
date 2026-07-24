#[repr(C)]
pub struct RnnFfiReadView {
    pub model_name_ptr: *const u8,
    pub model_name_len: usize,
    pub model_precision_ptr: *const u8,
    pub model_precision_len: usize,
    pub layer_meta_ptr: *const u8,
    pub layer_meta_len: usize,
    pub weights_ptr: *const u8,
    pub weights_len: usize,
    pub biases_ptr: *const u8,
    pub biases_len: usize,
    pub benchmark_ptr: *const u8,
    pub benchmark_len: usize,
    pub training_log_ptr: *const u8,
    pub training_log_len: usize,
    pub topology_len: usize,
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

#[no_mangle]
pub extern "C" fn rnn_ffi_build_f32(
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
    super::dispatch::build_f32(
        model_name_ptr, model_name_len,
        topology_ptr, topology_len,
        weights_ptr, weights_len,
        biases_ptr, biases_len,
        out_ptr, out_cap, out_used,
    )
}

#[no_mangle]
pub extern "C" fn rnn_ffi_build_f64(
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
    super::dispatch::build_f64(
        model_name_ptr, model_name_len,
        topology_ptr, topology_len,
        weights_ptr, weights_len,
        biases_ptr, biases_len,
        out_ptr, out_cap, out_used,
    )
}

#[no_mangle]
pub extern "C" fn rnn_ffi_train(
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
    super::dispatch::train(
        model_ptr, model_len,
        topology_ptr, topology_len,
        input_ptr, input_len,
        target_ptr, target_len,
        learning_rate, train_seconds,
        start_iteration, max_iterations,
        out_avg_loss, out_iterations,
    )
}

#[no_mangle]
pub extern "C" fn rnn_ffi_run(
    model_ptr: *mut u8,
    model_len: usize,
    topology_ptr: *const usize,
    topology_len: usize,
    input_ptr: *const f32,
    input_len: usize,
    out_score: *mut f64,
) -> i32 {
    super::dispatch::run(
        model_ptr, model_len,
        topology_ptr, topology_len,
        input_ptr, input_len,
        out_score,
    )
}

#[no_mangle]
pub extern "C" fn rnn_ffi_read_rnn(
    model_ptr: *mut u8,
    model_len: usize,
    device_id_ptr: *const u8,
    device_id_len: usize,
    out_topology_ptr: *mut usize,
    out_topology_cap: usize,
    out_view: *mut RnnFfiReadView,
) -> i32 {
    super::dispatch::read_rnn(
        model_ptr, model_len,
        device_id_ptr, device_id_len,
        out_topology_ptr, out_topology_cap,
        out_view,
    )
}
