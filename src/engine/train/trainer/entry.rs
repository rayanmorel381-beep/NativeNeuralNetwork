use crate::base::activations::ActivationKind;
use crate::engine::evolve::{evolve_add_layer, evolve_add_neuron, evolve_split_neuron};
use crate::format::model_format::BLOB_LAYER_META;
use crate::format::model_format::BLOB_WEIGHTS;
use crate::format::model_format::BLOB_BIASES;
use super::train_sgd::{evaluate_on_decrypted, sgd_on_decrypted, EvalAccum};
use super::fork::{sgd_pool_create, sgd_pool_start, sgd_pool_finish, sgd_pool_start_n, sgd_pool_finish_n, sgd_pool_destroy, SgdPoolConfig};
use crate::engine::runtime::{
    detect_hardware_profile, derive_runtime_plan, adjusted_samples_per_cycle,
    SwapBatch, ensure_hardware_init,
};
use crate::engine::rnn_flow::{
    blob_range, rnn_dtype,
    RnnFlowError, MAX_LAYERS, LAYER_META_ENTRY_SIZE,
};
use crate::format::model_config::{ModelConfig, ProgressCallback, ProgressUpdate, TrainingMetrics};

fn configure_sample_pipeline(_model_name: &str) {}

fn sample_pipeline_fill(base_iteration: usize, lane: usize, input: &mut [f32], target: &mut [f32]) {
    let seed = (base_iteration + lane + 1) as f32;
    for (idx, slot) in input.iter_mut().enumerate() {
        *slot = seed + idx as f32;
    }
    for (idx, slot) in target.iter_mut().enumerate() {
        *slot = seed + idx as f32;
    }
}

pub struct TrainLoopConfig {
    pub sample_filler: fn(usize, usize, &mut [f32], &mut [f32]),
    pub train_seconds: u64,
    pub max_iterations: Option<usize>,
    pub start_iteration: usize,
    pub elapsed_offset_ns: u64,
    pub progress_callback: Option<ProgressCallback>,
    pub progress_interval_ns: u64,
    pub policy: crate::format::model_config::ResourcePolicy,
}

pub struct TrainLoopResult {
    pub iterations: usize,
    pub avg_loss: f64,
    pub last_loss: f64,
    pub samples_per_cycle: usize,
    pub elapsed_ns: u64,
}

impl TrainLoopResult {
    pub fn effective_cycle_size(&self) -> usize {
        self.samples_per_cycle.max(1)
    }
}

pub struct ParsedLayout {
    pub(crate) dtype: u8,
    pub(crate) topology: [usize; MAX_LAYERS],
    pub(crate) topo_len: usize,
    pub(crate) w_start: usize,
    pub(crate) w_end: usize,
    pub(crate) b_start: usize,
    pub(crate) b_end: usize,
    pub(crate) hidden_activation: ActivationKind,
    pub(crate) output_activation: ActivationKind,
    pub(crate) loss: crate::engine::eval::losses::LossKind,
}

pub(crate) fn parse_layout(bytes: &[u8]) -> Result<ParsedLayout, RnnFlowError> {
    crate::format::rnn_format::validate(bytes, None)
        .map_err(|_| RnnFlowError::BadBytes)?;
    let dtype = rnn_dtype(bytes)?;
    let (lm_start, lm_end) = blob_range(bytes, BLOB_LAYER_META)?;
    let (w_start, w_end) = blob_range(bytes, BLOB_WEIGHTS)?;
    let (b_start, b_end) = blob_range(bytes, BLOB_BIASES)?;
    let lm_len = lm_end - lm_start;
    if lm_len < LAYER_META_ENTRY_SIZE || lm_len % LAYER_META_ENTRY_SIZE != 0 {
        return Err(RnnFlowError::BadBytes);
    }
    let layer_count = lm_len / LAYER_META_ENTRY_SIZE;
    if layer_count + 1 > MAX_LAYERS {
        return Err(RnnFlowError::BadBytes);
    }
    let mut topology = [0usize; MAX_LAYERS];
    let mut topo_len = 0usize;
    let mut hidden_activation = ActivationKind::Identity;
    let mut output_activation = ActivationKind::Identity;
    let mut loss_tag = 0u8;
    for idx in 0..layer_count {
        let base = lm_start + idx * LAYER_META_ENTRY_SIZE;
        let input_size = u32::from_le_bytes([
            bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3],
        ]) as usize;
        let output_size = u32::from_le_bytes([
            bytes[base + 4], bytes[base + 5], bytes[base + 6], bytes[base + 7],
        ]) as usize;
        let activation = ActivationKind::from_u8(bytes[base + 16])
            .ok_or(RnnFlowError::BadBytes)?;
        if idx == 0 {
            topology[0] = input_size;
            topo_len = 1;
            if layer_count > 1 { hidden_activation = activation; }
        } else {
            if topology[topo_len - 1] != input_size {
                return Err(RnnFlowError::BadBytes);
            }
            if idx < layer_count - 1 && activation != hidden_activation {
                return Err(RnnFlowError::BadBytes);
            }
        }
        topology[topo_len] = output_size;
        topo_len += 1;
        if idx == layer_count - 1 {
            output_activation = activation;
            loss_tag = bytes[base + 17];
        }
    }
    if topo_len < 2 {
        return Err(RnnFlowError::BadBytes);
    }
    Ok(ParsedLayout {
        dtype, topology, topo_len, w_start, w_end, b_start, b_end,
        hidden_activation, output_activation,
        loss: crate::engine::eval::losses::LossKind::from_tag(loss_tag, 1.0),
    })
}

pub fn train(
    bytes: &mut [u8],
    config: &ModelConfig,
) -> Result<(f64, usize), RnnFlowError> {
    ensure_hardware_init();

    let samples = config.samples;
    let learning_rate = config.learning_rate;
    let gradient_clip = config.gradient_clip;
    #[cfg(feature = "publisher-trust-service")]
    let current_device_id = config.device_id;
    #[cfg(feature = "publisher-trust-service")]
    let trusted_publisher_pubkeys = config.trusted_publisher_pubkeys;
    let evolve_out = config.evolve_out;
    let sample_filler = config.sample_filler;
    let train_seconds = config.train_seconds;
    let max_iterations = config.max_iterations;
    let start_iteration = config.start_iteration;
    let elapsed_offset_ns = config.elapsed_offset_ns;
    let progress_callback = config.progress_callback;
    let progress_interval_ns = config.progress_interval_ns;

    if sample_filler.is_some() {
        let result = train_loop_with_policy(bytes, config)?;
        let packed = pack_train_loop_result(&result);
        return Ok(packed);
    }

    if train_seconds > 0 && evolve_out.is_none() {
        configure_sample_pipeline(config.model_name);
        let cfg = TrainLoopConfig {
            sample_filler: sample_pipeline_fill,
            train_seconds,
            max_iterations,
            start_iteration,
            elapsed_offset_ns,
            progress_callback,
            progress_interval_ns,
            policy: config.policy,
        };
        let result = train_loop(bytes, learning_rate, gradient_clip, &cfg)?;
        let packed = pack_train_loop_result(&result);
        return Ok(packed);
    }

    if let Some((evolve_idx, iteration_seed)) = evolve_out {
        let enc_len = config.encrypted_data_len.unwrap_or(bytes.len());
        return execute_evolve(bytes, enc_len, evolve_idx, iteration_seed);
    }

    if let Some(evolve_cfg) = config.evolve {
        if train_seconds > 0 {
            return train_with_evolve(bytes, config, evolve_cfg);
        }
    }

    if samples.is_empty() {
        return Ok((0.0, 0));
    }
    let len = bytes.len();

    let plain_len = crate::security::crypto::decrypt_rnn(bytes, len);

    let layout = match parse_layout(&bytes[..plain_len]) {
        Ok(l) => l,
        Err(e) => { crate::security::crypto::encrypt_rnn(bytes, plain_len); return Err(e); }
    };
    #[cfg(feature = "publisher-trust-service")]
    crate::format::model_config::rnn::validate(crate::format::model_config::rnn::ValidateRequest::Distribution {
        bytes: &bytes[..plain_len],
        current_device_id,
        trusted_publisher_pubkeys,
    })
    .inspect_err(|_| {
        crate::security::crypto::encrypt_rnn(bytes, plain_len);
    })
    .map_err(|_| RnnFlowError::Model)?;

    let result = sgd_on_decrypted(bytes, samples, learning_rate, gradient_clip, &layout);

    if result.is_ok() {
        let hw = detect_hardware_profile();
        let plan = derive_runtime_plan(hw);
        let n_agents = plan.max_workers;
        if n_agents >= 2 && crate::engine::runtime::cache_ensure(n_agents, len) {
            if crate::engine::runtime::cache_full() {
                crate::engine::runtime::cache_reset_used();
            }
            crate::engine::runtime::cache_store(bytes);
            if crate::engine::runtime::cache_full() {
                match layout.dtype {
                    0 => crate::engine::runtime::compact_f32(layout.w_start, layout.w_end, layout.b_start, layout.b_end, bytes),
                    1 => crate::engine::runtime::compact_f64(layout.w_start, layout.w_end, layout.b_start, layout.b_end, bytes),
                    _ => {}
                }
            }
        }
    }

    crate::security::crypto::encrypt_rnn(bytes, plain_len);
    result
}

fn pack_train_loop_result(r: &TrainLoopResult) -> (f64, usize) {
    (r.avg_loss, r.iterations)
}

fn fill_swap_batch_engine(
    buf: &mut SwapBatch,
    base_iteration: usize,
    fill_count: usize,
    _max_workers: usize,
    filler: fn(usize, usize, &mut [f32], &mut [f32]),
) {
    let input_size = buf.input_size();
    let output_size = buf.output_size();
    let sample_bytes = buf.sample_bytes();
    let count = fill_count.min(buf.max_count());
    buf.set_count(count);
    let ptr = buf.ptr();

    for lane in 0..count {
        let offset = lane * sample_bytes;
        unsafe {
            let input = core::slice::from_raw_parts_mut(ptr.add(offset) as *mut f32, input_size);
            let target = core::slice::from_raw_parts_mut(ptr.add(offset + input_size * 4) as *mut f32, output_size);
            filler(base_iteration, lane, input, target);
        }
    }
}


fn train_loop(
    bytes: &mut [u8],
    learning_rate: f32,
    gradient_clip: Option<f32>,
    cfg: &TrainLoopConfig,
) -> Result<TrainLoopResult, RnnFlowError> {
    ensure_hardware_init();

    let len = bytes.len();
    let plain_len = crate::security::crypto::decrypt_rnn(bytes, len);
    let layout = match parse_layout(&bytes[..plain_len]) {
        Ok(l) => l,
        Err(e) => { crate::security::crypto::encrypt_rnn(bytes, plain_len); return Err(e); }
    };
    let topo_arr = layout.topology;
    let topo_len = layout.topo_len;
    let input_size = topo_arr[0];
    let output_size = topo_arr[topo_len - 1];

    let hw_profile = detect_hardware_profile();
    let plan = derive_runtime_plan(hw_profile);
    let samples_per_cycle = adjusted_samples_per_cycle(plan.batch_size, hw_profile);
    let chunk_per_worker = samples_per_cycle / plan.max_workers.max(1);

    let start_ns = crate::engine::runtime::hardware::monotonic_ns();
    let mut iterations = cfg.start_iteration;
    let mut run_iterations = 0usize;
    let mut loss_acc = crate::engine::eval::metrics::RunningMeanF64::new();
    let mut last_loss = 0.0f64;
    let target_iterations = cfg.max_iterations.unwrap_or(usize::MAX);
    let time_limited = cfg.train_seconds > 0;
    let train_ns = cfg.train_seconds.saturating_mul(1_000_000_000);

    let filler = cfg.sample_filler;
    let progress_cb = cfg.progress_callback;
    let progress_interval = if cfg.progress_interval_ns > 0 { cfg.progress_interval_ns } else { 1_000_000_000 };
    let mut last_progress_ns = start_ns.saturating_sub(progress_interval);

    let logical = hw_profile.cores.max(1);
    let initial_metrics = TrainingMetrics {
        iterations: cfg.start_iteration,
        avg_loss: 0.0,
        last_loss: 0.0,
        elapsed_ns: 0,
        cpu_temp: None,
        logical_cores: logical,
        current_workers: 0,
        max_workers: plan.max_workers,
        eval_mse: 0.0,
        eval_mae: 0.0,
        eval_accuracy: 0.0,
        eval_cross_entropy: 0.0,
        eval_confidence: 0.0,
        eval_ops_per_second: 0.0,
        eval_bytes_per_second: 0.0,
        eval_arithmetic_intensity: 0.0,
        eval_memory_heavy: false,
    };
    let desired_cpu = (cfg.policy)(&initial_metrics).clamp(0.05, 1.0);
    let worker_cores = ((logical as f64 * desired_cpu) as usize).clamp(1, logical.min(plan.max_workers).max(1));
    let core_mask: usize = (1usize << worker_cores) - 1;

    let pool_cfg = SgdPoolConfig {
        learning_rate, gradient_clip, filler, input_size, output_size,
        chunk_size: chunk_per_worker, core_mask,
    };
    let pool = sgd_pool_create(worker_cores, len, &layout, &pool_cfg, bytes);

    if pool.is_active() {
        crate::engine::runtime::hardware::set_affinity(core_mask);

        while iterations < target_iterations && (!time_limited || (crate::engine::runtime::hardware::monotonic_ns() - start_ns) < train_ns) {
            sgd_pool_start(&pool, iterations, chunk_per_worker);
            let (batch_loss, batch_count) = sgd_pool_finish(&pool, bytes, &layout)?;
            if batch_count > 0 { last_loss = batch_loss / batch_count as f64; }
            loss_acc.merge(crate::engine::eval::metrics::RunningMeanF64 { sum: batch_loss, count: batch_count as u64 });
            iterations = iterations.saturating_add(batch_count);
            run_iterations = run_iterations.saturating_add(batch_count);

            if let Some(cb) = progress_cb {
                let cb_now = crate::engine::runtime::hardware::monotonic_ns();
                if cb_now.saturating_sub(last_progress_ns) >= progress_interval && run_iterations > 0 {
                    let avg = loss_acc.value().unwrap_or(0.0);
                    let total_ns = cb_now.saturating_sub(start_ns).saturating_add(cfg.elapsed_offset_ns);
                    cb(&ProgressUpdate {
                        iterations,
                        elapsed_ns: total_ns,
                        avg_loss: avg,
                        last_loss,
                        plain_len,
                        topology_ptr: topo_arr.as_ptr(),
                        topology_len: topo_len,
                        samples_per_cycle,
                    });
                    last_progress_ns = cb_now;
                }
            }
        }
    } else {
        let mut batch = SwapBatch::new(input_size, output_size, samples_per_cycle)
            .ok_or(RnnFlowError::CapacityTooSmall)?;

        let ref_entry_size = core::mem::size_of::<(&[f32], &[f32])>();
        let ref_buf_bytes = samples_per_cycle.saturating_mul(ref_entry_size);
        let ref_buf = crate::engine::runtime::hardware::mmap_shared_anon(ref_buf_bytes);
        if ref_buf.is_null() {
            batch.release();
            return Err(RnnFlowError::CapacityTooSmall);
        }

        while iterations < target_iterations && (!time_limited || (crate::engine::runtime::hardware::monotonic_ns() - start_ns) < train_ns) {
            let n = samples_per_cycle.min(target_iterations.saturating_sub(iterations));

            fill_swap_batch_engine(&mut batch, iterations, n, plan.max_workers, filler);

            for i in 0..n {
                let (inp, tgt) = batch.get_sample(i);
                unsafe {
                    let slot = ref_buf.add(i * ref_entry_size) as *mut (&[f32], &[f32]);
                    core::ptr::write(slot, (inp, tgt));
                }
            }

            let sample_refs = unsafe { core::slice::from_raw_parts(ref_buf as *const (&[f32], &[f32]), n) };
            let (batch_loss, batch_count) = sgd_on_decrypted(bytes, sample_refs, learning_rate, gradient_clip, &layout)?;
            if batch_count > 0 { last_loss = batch_loss / batch_count as f64; }
            loss_acc.merge(crate::engine::eval::metrics::RunningMeanF64 { sum: batch_loss, count: batch_count as u64 });
            iterations = iterations.saturating_add(batch_count);
            run_iterations = run_iterations.saturating_add(batch_count);

            let now_ns = crate::engine::runtime::hardware::monotonic_ns();
            if let Some(cb) = progress_cb {
                if now_ns.saturating_sub(last_progress_ns) >= progress_interval && run_iterations > 0 {
                    let avg = loss_acc.value().unwrap_or(0.0);
                    let total_ns = now_ns.saturating_sub(start_ns).saturating_add(cfg.elapsed_offset_ns);
                    cb(&ProgressUpdate {
                        iterations,
                        elapsed_ns: total_ns,
                        avg_loss: avg,
                        last_loss,
                        plain_len,
                        topology_ptr: topo_arr.as_ptr(),
                        topology_len: topo_len,
                        samples_per_cycle,
                    });
                    last_progress_ns = now_ns;
                }
            }
        }

        crate::engine::runtime::hardware::munmap(ref_buf, ref_buf_bytes);
        batch.release();
    }

    sgd_pool_destroy(&pool);

    if worker_cores >= 2 && crate::engine::runtime::cache_ensure(worker_cores, len) {
        crate::engine::runtime::cache_store(bytes);
        if crate::engine::runtime::cache_full() {
            match layout.dtype {
                0 => crate::engine::runtime::compact_f32(layout.w_start, layout.w_end, layout.b_start, layout.b_end, bytes),
                1 => crate::engine::runtime::compact_f64(layout.w_start, layout.w_end, layout.b_start, layout.b_end, bytes),
                _ => {}
            }
            crate::engine::runtime::cache_release();
        }
    }

    crate::security::crypto::encrypt_rnn(bytes, plain_len);

    let elapsed_ns = crate::engine::runtime::hardware::monotonic_ns().saturating_sub(start_ns)
        .saturating_add(cfg.elapsed_offset_ns);
    let avg_loss = loss_acc.value().unwrap_or(0.0);
    let mut result = TrainLoopResult { iterations, avg_loss, last_loss, samples_per_cycle, elapsed_ns };
    result.samples_per_cycle = result.effective_cycle_size();

    Ok(result)
}

pub fn train_loop_with_policy(
    bytes: &mut [u8],
    rc: &crate::format::model_config::ModelConfig,
) -> Result<TrainLoopResult, RnnFlowError> {
    ensure_hardware_init();

    let filler = match rc.sample_filler {
        Some(f) => f,
        None => return Err(RnnFlowError::InvalidTopology),
    };

    let len = bytes.len();
    let plain_len = crate::security::crypto::decrypt_rnn(bytes, len);
    let layout = match parse_layout(&bytes[..plain_len]) {
        Ok(l) => l,
        Err(e) => { crate::security::crypto::encrypt_rnn(bytes, plain_len); return Err(e); }
    };
    let topo_arr = layout.topology;
    let topo_len = layout.topo_len;
    let input_size = topo_arr[0];
    let output_size = topo_arr[topo_len - 1];

    let hw_profile = detect_hardware_profile();
    let plan = derive_runtime_plan(hw_profile);
    let samples_per_cycle = adjusted_samples_per_cycle(plan.batch_size, hw_profile);
    let logical = hw_profile.cores.max(1);
    let max_workers = plan.max_workers;
    let chunk_per_worker = samples_per_cycle / max_workers.max(1);
    let core_mask: usize = (1usize << max_workers) - 1;

    let start_ns = crate::engine::runtime::hardware::monotonic_ns();
    let mut iterations = rc.start_iteration;
    let mut run_iterations = 0usize;
    let mut loss_acc = crate::engine::eval::metrics::RunningMeanF64::new();
    let mut last_loss = 0.0f64;
    let target_iterations = rc.max_iterations.unwrap_or(usize::MAX);
    let time_limited = rc.train_seconds > 0;
    let train_ns = rc.train_seconds.saturating_mul(1_000_000_000);
    let progress_cb = rc.progress_callback;
    let progress_interval = if rc.progress_interval_ns > 0 { rc.progress_interval_ns } else { 1_000_000_000 };
    let mut last_progress_ns = start_ns.saturating_sub(progress_interval);
    let mut active_workers = max_workers;
    let policy: crate::format::model_config::ResourcePolicy = rc.policy;

    let mut eval_total = EvalAccum::new();
    let mut eval_batch = EvalAccum::new();
    let eval_count = chunk_per_worker.max(1);
    let mut eval_swap = SwapBatch::new(input_size, output_size, eval_count);
    let eval_ref_entry = core::mem::size_of::<(&[f32], &[f32])>();
    let eval_ref_bytes = eval_count.saturating_mul(eval_ref_entry);
    let eval_ref_buf = crate::engine::runtime::hardware::mmap_shared_anon(eval_ref_bytes);
    let mut last_eval_ns = start_ns.saturating_sub(progress_interval);

    let pool_cfg = SgdPoolConfig {
        learning_rate: rc.learning_rate, gradient_clip: rc.gradient_clip,
        filler, input_size, output_size,
        chunk_size: chunk_per_worker, core_mask,
    };
    let pool = sgd_pool_create(max_workers, len, &layout, &pool_cfg, bytes);

    if pool.is_active() {
        crate::engine::runtime::hardware::set_affinity(core_mask);

        while iterations < target_iterations && (!time_limited || (crate::engine::runtime::hardware::monotonic_ns() - start_ns) < train_ns) {
            let now_ns = crate::engine::runtime::hardware::monotonic_ns();
            let elapsed = now_ns.saturating_sub(start_ns);
            let eval_secs = elapsed as f32 / 1_000_000_000.0;
            let has_eval_work = crate::observability::profiler::has_recorded_work(&eval_total.ops);
            let metrics = TrainingMetrics {
                iterations,
                avg_loss: loss_acc.value().unwrap_or(0.0),
                last_loss,
                elapsed_ns: elapsed,
                cpu_temp: None,
                logical_cores: logical,
                current_workers: active_workers,
                max_workers,
                eval_mse: eval_total.mse.value().unwrap_or(0.0) as f64,
                eval_mae: eval_total.mae.value().unwrap_or(0.0) as f64,
                eval_accuracy: eval_total.accuracy.value().unwrap_or(0.0) as f64,
                eval_cross_entropy: eval_total.cross_entropy.value().unwrap_or(0.0) as f64,
                eval_confidence: eval_total.confidence.value().unwrap_or(0.0) as f64,
                eval_ops_per_second: if has_eval_work { crate::observability::profiler::ops_per_second(&eval_total.ops, eval_secs) as f64 } else { 0.0 },
                eval_bytes_per_second: if has_eval_work { crate::observability::profiler::bytes_per_second(&eval_total.ops, eval_secs) as f64 } else { 0.0 },
                eval_arithmetic_intensity: crate::observability::profiler::arithmetic_intensity(&eval_total.ops) as f64,
                eval_memory_heavy: crate::observability::profiler::is_memory_heavy(&eval_total.ops),
            };
            let desired_cpu = policy(&metrics).clamp(0.05, 1.0);
            active_workers = ((logical as f64 * desired_cpu) as usize).clamp(1, max_workers);

            sgd_pool_start_n(&pool, iterations, chunk_per_worker, active_workers);
            let (batch_loss, batch_count) = sgd_pool_finish_n(&pool, bytes, &layout, active_workers)?;
            if batch_count > 0 { last_loss = batch_loss / batch_count as f64; }
            loss_acc.merge(crate::engine::eval::metrics::RunningMeanF64 { sum: batch_loss, count: batch_count as u64 });
            iterations = iterations.saturating_add(batch_count);
            run_iterations = run_iterations.saturating_add(batch_count);

            let ev_now = crate::engine::runtime::hardware::monotonic_ns();
            if ev_now.saturating_sub(last_eval_ns) >= progress_interval && !eval_ref_buf.is_null() {
                if let Some(swap) = eval_swap.as_mut() {
                    fill_swap_batch_engine(swap, iterations, eval_count, plan.max_workers, filler);
                    for i in 0..eval_count {
                        let (inp, tgt) = swap.get_sample(i);
                        unsafe {
                            let slot = eval_ref_buf.add(i * eval_ref_entry) as *mut (&[f32], &[f32]);
                            core::ptr::write(slot, (inp, tgt));
                        }
                    }
                    let refs = unsafe { core::slice::from_raw_parts(eval_ref_buf as *const (&[f32], &[f32]), eval_count) };
                    eval_batch.reset();
                    if evaluate_on_decrypted(bytes, refs, &layout, &mut eval_batch).is_ok() {
                        eval_total.merge(&eval_batch);
                    }
                    last_eval_ns = ev_now;
                }
            }

            if let Some(cb) = progress_cb {
                let cb_now = crate::engine::runtime::hardware::monotonic_ns();
                if cb_now.saturating_sub(last_progress_ns) >= progress_interval && run_iterations > 0 {
                    let avg = loss_acc.value().unwrap_or(0.0);
                    let total_ns = cb_now.saturating_sub(start_ns).saturating_add(rc.elapsed_offset_ns);
                    cb(&ProgressUpdate {
                        iterations,
                        elapsed_ns: total_ns,
                        avg_loss: avg,
                        last_loss,
                        plain_len,
                        topology_ptr: topo_arr.as_ptr(),
                        topology_len: topo_len,
                        samples_per_cycle,
                    });
                    last_progress_ns = cb_now;
                }
            }
        }
    }

    sgd_pool_destroy(&pool);

    if !eval_ref_buf.is_null() {
        crate::engine::runtime::hardware::munmap(eval_ref_buf, eval_ref_bytes);
    }
    if let Some(mut swap) = eval_swap {
        swap.release();
    }

    let n_agents = max_workers;
    if n_agents >= 2 && crate::engine::runtime::cache_ensure(n_agents, len) {
        if crate::engine::runtime::cache_full() {
            crate::engine::runtime::cache_reset_used();
        }
        crate::engine::runtime::cache_store(bytes);
        if crate::engine::runtime::cache_full() {
            match layout.dtype {
                0 => crate::engine::runtime::compact_f32(layout.w_start, layout.w_end, layout.b_start, layout.b_end, bytes),
                1 => crate::engine::runtime::compact_f64(layout.w_start, layout.w_end, layout.b_start, layout.b_end, bytes),
                _ => {}
            }
        }
    }

    crate::security::crypto::encrypt_rnn(bytes, plain_len);

    let elapsed_ns = crate::engine::runtime::hardware::monotonic_ns().saturating_sub(start_ns)
        .saturating_add(rc.elapsed_offset_ns);
    let avg_loss = loss_acc.value().unwrap_or(0.0);
    let mut result = TrainLoopResult { iterations, avg_loss, last_loss, samples_per_cycle, elapsed_ns };
    result.samples_per_cycle = result.effective_cycle_size();

    Ok(result)
}

pub(crate) fn execute_evolve(
    bytes: &mut [u8],
    enc_len: usize,
    evolve_idx: usize,
    iteration_seed: usize,
) -> Result<(f64, usize), RnnFlowError> {
    apply_evolve_step(bytes, enc_len, evolve_idx, iteration_seed)
}

pub(crate) fn apply_evolve_step(
    bytes: &mut [u8],
    enc_len: usize,
    evolve_idx: usize,
    iteration_seed: usize,
) -> Result<(f64, usize), RnnFlowError> {
    let hw = detect_hardware_profile();
    let plan = derive_runtime_plan(hw);
    let allowed = plan.max_workers.max(1);
    let mask: usize = (1usize << allowed) - 1;
    crate::engine::runtime::hardware::set_affinity(mask);

    let len = bytes.len();
    let src_ptr = crate::engine::runtime::hardware::mmap_shared_anon(len);
    if src_ptr.is_null() {
        return Err(RnnFlowError::CapacityTooSmall);
    }
    let src_buf = unsafe { core::slice::from_raw_parts_mut(src_ptr, len) };
    src_buf[..enc_len].copy_from_slice(&bytes[..enc_len]);
    let src_plain_len = crate::security::crypto::decrypt_rnn(src_buf, enc_len);

    let layout = match parse_layout(&src_buf[..src_plain_len]) {
        Ok(l) => l,
        Err(e) => {
            crate::engine::runtime::hardware::munmap(src_ptr, len);
            return Err(e);
        }
    };
    let hidden_count = layout.topo_len.saturating_sub(2);
    if hidden_count == 0 {
        crate::engine::runtime::hardware::munmap(src_ptr, len);
        return Err(RnnFlowError::InvalidTopology);
    }

    let seed = iteration_seed as u64 ^ 0xA1B2_C3D4_E5F6_0718;
    let target_layer = 1 + (evolve_idx % hidden_count);

    let param_bytes = (layout.w_end - layout.w_start) + (layout.b_end - layout.b_start);
    let evolve_out_size = param_bytes.saturating_mul(4).max(len);

    let evo_ptr = crate::engine::runtime::hardware::mmap_shared_anon(evolve_out_size);
    if evo_ptr.is_null() {
        crate::engine::runtime::hardware::munmap(src_ptr, len);
        return Err(RnnFlowError::CapacityTooSmall);
    }
    let evo_buf = unsafe { core::slice::from_raw_parts_mut(evo_ptr, evolve_out_size) };

    let src = &src_buf[..src_plain_len];
    let result = match evolve_idx % 3 {
        1 => {
            let neuron_idx = layout.topology[target_layer] / 2;
            evolve_split_neuron(src, target_layer, neuron_idx, seed, evo_buf)
        }
        2 if layout.topo_len < MAX_LAYERS => {
            let new_size = layout.topology[target_layer];
            evolve_add_layer(src, target_layer, new_size, seed, evo_buf)
        }
        _ => evolve_add_neuron(src, target_layer, seed, evo_buf),
    };

    crate::engine::runtime::hardware::munmap(src_ptr, len);

    match result {
        Ok(new_len) => {
            let enc_total = crate::security::crypto::encrypted_rnn_size(new_len).unwrap_or(usize::MAX);
            if enc_total > bytes.len() {
                crate::engine::runtime::hardware::munmap(evo_ptr, evolve_out_size);
                return Err(RnnFlowError::CapacityTooSmall);
            }
            bytes[..new_len].copy_from_slice(&evo_buf[..new_len]);
            crate::engine::runtime::hardware::munmap(evo_ptr, evolve_out_size);
            let enc_new = crate::security::crypto::encrypt_rnn(bytes, new_len);
            Ok((0.0, enc_new))
        }
        Err(e) => {
            crate::engine::runtime::hardware::munmap(evo_ptr, evolve_out_size);
            Err(e)
        }
    }
}

fn train_with_evolve(
    bytes: &mut [u8],
    config: &ModelConfig,
    evolve_cfg: crate::engine::evolve::EvolveConfig,
) -> Result<(f64, usize), RnnFlowError> {
    use crate::engine::evolve::{PhaseScheduler, PhaseEvent, PhaseResult};

    configure_sample_pipeline(config.model_name);

    let mut sched = PhaseScheduler::new(
        evolve_cfg,
        config.train_seconds,
        config.max_iterations,
        config.start_iteration,
        config.elapsed_offset_ns,
    );
    let mut current_enc_len = config.encrypted_data_len.unwrap_or(bytes.len());

    loop {
        match sched.next_event() {
            PhaseEvent::Phase(plan) => {
                let phase_idx = plan.phase_idx;
                let cfg = TrainLoopConfig {
                    sample_filler: sample_pipeline_fill,
                    train_seconds: plan.phase_seconds,
                    max_iterations: plan.phase_max_iterations,
                    start_iteration: plan.accumulated_iterations,
                    elapsed_offset_ns: plan.accumulated_elapsed_ns,
                    progress_callback: config.progress_callback,
                    progress_interval_ns: config.progress_interval_ns,
                    policy: config.policy,
                };
                let result = train_loop(bytes, config.learning_rate, config.gradient_clip, &cfg)?;
                sched.record(&PhaseResult {
                    iterations: result.iterations,
                    elapsed_ns: result.elapsed_ns,
                    avg_loss: result.avg_loss,
                    last_loss: result.last_loss,
                });
                if let Some(cb) = config.progress_callback {
                    let update = ProgressUpdate {
                        iterations: sched.final_iterations(),
                        elapsed_ns: sched.final_elapsed_ns(),
                        avg_loss: sched.final_avg_loss(),
                        last_loss: sched.final_last_loss(),
                        plain_len: current_enc_len,
                        topology_ptr: core::ptr::null(),
                        topology_len: 0,
                        samples_per_cycle: phase_idx,
                    };
                    cb(&update);
                }
            }
            PhaseEvent::Evolve { evolve_idx, iteration_seed } => {
                let new_enc_len = crate::engine::evolve::do_evolve(
                    bytes,
                    current_enc_len,
                    evolve_idx,
                    iteration_seed,
                )
                .map_err(|_| RnnFlowError::Model)?;
                current_enc_len = new_enc_len;
            }
            PhaseEvent::Done => break,
        }
    }

    if let Some(cb) = config.progress_callback {
        let update = ProgressUpdate {
            iterations: sched.final_iterations(),
            elapsed_ns: sched.final_elapsed_ns(),
            avg_loss: sched.final_avg_loss(),
            last_loss: sched.final_last_loss(),
            plain_len: current_enc_len,
            topology_ptr: core::ptr::null(),
            topology_len: 0,
            samples_per_cycle: sched.evolves_done(),
        };
        cb(&update);
    }

    Ok((sched.final_avg_loss(), current_enc_len))
}

