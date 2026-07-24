use crate::format::model_config::ModelConfig;
use crate::api::rnn_api::core_api::RnnApiError;
use crate::engine::train::trainer::ParsedLayout;

const RESULT_ENTRY: usize = 16;
const ENSEMBLE_MAX_OUT: usize = 4096;

struct InferJob {
    chunk: usize,
    n_inputs: usize,
    input_size: usize,
    output_size: usize,
}

pub fn run_mlp(bytes: &[u8], config: &mut ModelConfig) -> Result<(f64, usize), RnnApiError> {
    let hw = crate::engine::runtime::detect_hardware_profile();
    let plan = crate::engine::runtime::derive_runtime_plan(hw);
    let allowed = plan.max_workers.max(1);
    let mask: usize = (1usize << allowed) - 1;
    crate::engine::runtime::hardware::set_affinity(mask);

    let len = bytes.len();
    let tmp_ptr = crate::engine::runtime::hardware::mmap_shared_anon(len);
    if tmp_ptr.is_null() { return Err(RnnApiError::CapacityTooSmall); }
    let tmp = unsafe { core::slice::from_raw_parts_mut(tmp_ptr, len) };
    tmp.copy_from_slice(bytes);
    let plain_len = crate::security::crypto::decrypt_rnn(tmp, len);

    let layout = match crate::engine::train::trainer::parse_layout(&tmp[..plain_len]) {
        Ok(l) => l,
        Err(_) => {
            crate::engine::runtime::hardware::munmap(tmp_ptr, len);
            return Err(RnnApiError::BadBytes);
        }
    };

    let topo = &layout.topology[..layout.topo_len];
    let input_size = topo[0];
    let output_size = topo[layout.topo_len - 1];

    match &mut config.precision {
        crate::format::model_config::Precision::InferF32 { runtime_input, runtime_output } => {
            let r = run_infer_f32(tmp, &layout, input_size, output_size, runtime_input, runtime_output);
            crate::engine::runtime::hardware::munmap(tmp_ptr, len);
            return r;
        }
        crate::format::model_config::Precision::InferF64 { runtime_input, runtime_output } => {
            let r = run_infer_f64(tmp, &layout, input_size, output_size, runtime_input, runtime_output);
            crate::engine::runtime::hardware::munmap(tmp_ptr, len);
            return r;
        }
        _ => {}
    }

    let n_inputs = count_inputs(config, input_size);
    if n_inputs == 0 {
        crate::engine::runtime::hardware::munmap(tmp_ptr, len);
        return Err(RnnApiError::BadBytes);
    }

    let n_workers = allowed.min(n_inputs).max(1);

    crate::base::precision::set_precision(match layout.dtype {
        1 => crate::base::precision::Precision::F64,
        _ => crate::base::precision::Precision::F32,
    });

    let result = if n_workers < 2 {
        let input = sample_input(config, 0);
        match layout.dtype {
            0 => crate::engine::single_forward_f32(tmp, &layout, output_size, input, None),
            1 => crate::engine::single_forward_f64(tmp, &layout, input_size, output_size, input, None, None),
            _ => Err(RnnApiError::FormatMismatch),
        }
    } else {
        batch_forward(tmp, &layout, config, input_size, output_size, n_inputs, n_workers)
    };

    crate::engine::runtime::hardware::munmap(tmp_ptr, len);
    result
}

fn run_infer_f32(
    tmp: &[u8],
    layout: &ParsedLayout,
    input_size: usize,
    output_size: usize,
    input: &[f32],
    out: &mut [f32],
) -> Result<(f64, usize), RnnApiError> {
    if layout.dtype != 0 {
        return Err(RnnApiError::FormatMismatch);
    }
    let want_in = match crate::engine::rnn_flow::blob_range(tmp, crate::format::model_format::BLOB_CONV_SPEC) {
        Ok((s, e)) => crate::graph::conv::conv_net::conv_net_input_len(&tmp[s..e]).map_err(|_| RnnApiError::BadBytes)?,
        Err(_) => input_size,
    };
    if input.len() != want_in {
        return Err(RnnApiError::BadBytes);
    }
    if out.len() < output_size {
        return Err(RnnApiError::CapacityTooSmall);
    }
    crate::base::precision::set_precision(crate::base::precision::Precision::F32);
    let replicas = crate::engine::runtime::cache_replica_count(tmp.len());
    if replicas >= 2 {
        return ensemble_infer_f32(tmp.len(), layout, output_size, input, out, replicas);
    }
    crate::engine::single_forward_f32(tmp, layout, output_size, input, Some(out))
}

fn ensemble_infer_f32(
    model_len: usize,
    layout: &ParsedLayout,
    output_size: usize,
    input: &[f32],
    out: &mut [f32],
    replicas: usize,
) -> Result<(f64, usize), RnnApiError> {
    if output_size > ENSEMBLE_MAX_OUT {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let mut scratch_buf = [0.0f32; ENSEMBLE_MAX_OUT];
    let scratch = &mut scratch_buf[..output_size];
    for v in out.iter_mut().take(output_size) {
        *v = 0.0;
    }
    let mut counted = 0usize;
    for i in 0..replicas {
        let replica = match crate::engine::runtime::cache_replica_slice(i, model_len) {
            Some(r) => r,
            None => continue,
        };
        if crate::engine::single_forward_f32(replica, layout, output_size, input, Some(scratch)).is_ok() {
            for k in 0..output_size {
                out[k] += scratch[k];
            }
            counted += 1;
        }
    }
    if counted == 0 {
        return Err(RnnApiError::Model);
    }
    let inv = 1.0f32 / counted as f32;
    for v in out.iter_mut().take(output_size) {
        *v *= inv;
    }
    Ok((out[0] as f64, output_size))
}

fn run_infer_f64(
    tmp: &[u8],
    layout: &ParsedLayout,
    input_size: usize,
    output_size: usize,
    input: &[f64],
    out: &mut [f64],
) -> Result<(f64, usize), RnnApiError> {
    if layout.dtype != 1 {
        return Err(RnnApiError::FormatMismatch);
    }
    if input.len() != input_size {
        return Err(RnnApiError::BadBytes);
    }
    if out.len() < output_size {
        return Err(RnnApiError::CapacityTooSmall);
    }
    crate::base::precision::set_precision(crate::base::precision::Precision::F64);
    let replicas = crate::engine::runtime::cache_replica_count(tmp.len());
    if replicas >= 2 {
        return ensemble_infer_f64(tmp.len(), layout, input_size, output_size, input, out, replicas);
    }
    crate::engine::single_forward_f64(tmp, layout, input_size, output_size, &[], Some(input), Some(out))
}

fn ensemble_infer_f64(
    model_len: usize,
    layout: &ParsedLayout,
    input_size: usize,
    output_size: usize,
    input: &[f64],
    out: &mut [f64],
    replicas: usize,
) -> Result<(f64, usize), RnnApiError> {
    if output_size > ENSEMBLE_MAX_OUT {
        return Err(RnnApiError::CapacityTooSmall);
    }
    let mut scratch_buf = [0.0f64; ENSEMBLE_MAX_OUT];
    let scratch = &mut scratch_buf[..output_size];
    for v in out.iter_mut().take(output_size) {
        *v = 0.0;
    }
    let mut counted = 0usize;
    for i in 0..replicas {
        let replica = match crate::engine::runtime::cache_replica_slice(i, model_len) {
            Some(r) => r,
            None => continue,
        };
        if crate::engine::single_forward_f64(replica, layout, input_size, output_size, &[], Some(input), Some(scratch)).is_ok() {
            for k in 0..output_size {
                out[k] += scratch[k];
            }
            counted += 1;
        }
    }
    if counted == 0 {
        return Err(RnnApiError::Model);
    }
    let inv = 1.0f64 / counted as f64;
    for v in out.iter_mut().take(output_size) {
        *v *= inv;
    }
    Ok((out[0], output_size))
}

fn count_inputs(config: &ModelConfig, input_size: usize) -> usize {
    match config.precision {
        crate::format::model_config::Precision::F32 { runtime_input: Some(ri), .. } => {
            if ri.len() == input_size { 1 } else { 0 }
        }
        _ => {
            let n = config.samples.len();
            if n == 0 { return 0; }
            let mut i = 0;
            while i < n {
                if config.samples[i].0.len() != input_size { return 0; }
                i += 1;
            }
            n
        }
    }
}

fn sample_input<'a>(config: &'a ModelConfig, idx: usize) -> &'a [f32] {
    match config.precision {
        crate::format::model_config::Precision::F32 { runtime_input: Some(ri), .. } => ri,
        _ => config.samples[idx].0,
    }
}

fn batch_forward(
    tmp: &[u8],
    layout: &ParsedLayout,
    config: &ModelConfig,
    input_size: usize,
    output_size: usize,
    n_inputs: usize,
    n_workers: usize,
) -> Result<(f64, usize), RnnApiError> {
    let arena_size = n_workers * RESULT_ENTRY;
    let arena_ptr = crate::engine::runtime::hardware::mmap_shared_anon(arena_size);
    if arena_ptr.is_null() { return Err(RnnApiError::CapacityTooSmall); }
    unsafe { core::ptr::write_bytes(arena_ptr, 0, arena_size); }

    let chunk = n_inputs.div_ceil(n_workers);
    let infer_job = InferJob { chunk, n_inputs, input_size, output_size };
    let mut pids = [0i64; 12];
    let mut spawned = 0usize;

    for w in 0..n_workers {
        let pid = crate::engine::runtime::hardware::fork();
        if pid < 0 {
            for pid in pids.iter().take(spawned) { crate::engine::runtime::hardware::waitpid(*pid); }
            crate::engine::runtime::hardware::munmap(arena_ptr, arena_size);
            return Err(RnnApiError::CapacityTooSmall);
        }
        if pid == 0 {
            infer_worker(w, &infer_job, tmp, layout, config, arena_ptr);
        }
        pids[w] = pid;
        spawned += 1;
    }

    for pid in pids.iter().take(spawned) { crate::engine::runtime::hardware::waitpid(*pid); }

    let mut total_sum = 0.0f64;
    let mut total_count = 0u64;
    for w in 0..spawned {
        let slot = unsafe { arena_ptr.add(w * RESULT_ENTRY) };
        let s = f64::from_le_bytes(unsafe {
            [*slot, *slot.add(1), *slot.add(2), *slot.add(3),
             *slot.add(4), *slot.add(5), *slot.add(6), *slot.add(7)]
        });
        let c = u64::from_le_bytes(unsafe {
            [*slot.add(8), *slot.add(9), *slot.add(10), *slot.add(11),
             *slot.add(12), *slot.add(13), *slot.add(14), *slot.add(15)]
        });
        total_sum += s;
        total_count += c;
    }

    crate::engine::runtime::hardware::munmap(arena_ptr, arena_size);
    if total_count == 0 { return Err(RnnApiError::Model); }
    Ok((total_sum / total_count as f64, output_size))
}

fn infer_worker(
    worker_idx: usize,
    job: &InferJob,
    tmp: &[u8],
    layout: &ParsedLayout,
    config: &ModelConfig,
    arena_ptr: *mut u8,
) -> ! {
    let start = worker_idx * job.chunk;
    let end = (start + job.chunk).min(job.n_inputs);
    let mut sum = 0.0f64;
    let mut count = 0u64;

    for idx in start..end {
        let input = sample_input(config, idx);
        let res = match layout.dtype {
            0 => crate::engine::single_forward_f32(tmp, layout, job.output_size, input, None),
            1 => crate::engine::single_forward_f64(tmp, layout, job.input_size, job.output_size, input, None, None),
            _ => Err(RnnApiError::FormatMismatch),
        };
        if let Ok((val, _)) = res {
            sum += val;
            count += 1;
        }
    }

    let slot = unsafe { arena_ptr.add(worker_idx * RESULT_ENTRY) };
    unsafe {
        core::ptr::copy_nonoverlapping(sum.to_le_bytes().as_ptr(), slot, 8);
        core::ptr::copy_nonoverlapping(count.to_le_bytes().as_ptr(), slot.add(8), 8);
    }
    crate::engine::runtime::hardware::exit(0)
}
