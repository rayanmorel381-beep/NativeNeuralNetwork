use crate::api::rnn_api::core_api::RnnApiError;
use crate::engine::runtime::hardware;
use crate::text::tokenizer::{encode_with_bpe, parse_merges_blob, parse_vocab_blob, MergePair, Vocab};

const MAX_DATASET_DIRS: usize = 32;
const TRAIN_STEPS_PER_ITER: usize = 200;
const TRAIN_CORPUS_CAP: usize = 32 * 1024 * 1024;

struct Mapping {
    ptr: *mut u8,
    size: usize,
}

impl Mapping {
    fn alloc(size: usize) -> Option<Mapping> {
        if size == 0 {
            return Some(Mapping {
                ptr: core::ptr::NonNull::dangling().as_ptr(),
                size: 0,
            });
        }
        let ptr = hardware::mmap_shared_anon(size);
        if ptr.is_null() {
            None
        } else {
            Some(Mapping { ptr, size })
        }
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    fn f32_mut(&mut self) -> &mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut f32, self.size / 4) }
    }

    fn free(self) {
        if self.size != 0 {
            hardware::munmap(self.ptr, self.size);
        }
    }
}

pub fn train_lm(
    model_bytes: &mut [u8],
    train_seconds: u64,
    out: &mut [u8],
) -> Result<usize, RnnApiError> {
    if model_bytes.is_empty() {
        return Err(RnnApiError::NotInitialized);
    }

    let len = model_bytes.len();
    let mut tmp = Mapping::alloc(len).ok_or(RnnApiError::CapacityTooSmall)?;
    crate::engine::runtime::parallel_memcpy(tmp.bytes_mut(), model_bytes);
    let plain_len = crate::security::crypto::decrypt_rnn(tmp.bytes_mut(), len);


    let mut scratch_buf = [0u8; 16384];
    let mut scratch = crate::base::scratch::Scratch::new(&mut scratch_buf);
    let plain = unsafe { core::slice::from_raw_parts(tmp.ptr, plain_len) };
    let handle = match crate::format::rnn_format::parser::parse_rnn_from_bytes(plain, &mut scratch) {
        Ok(h) => h,
        Err(_) => {
            tmp.free();
            return Err(RnnApiError::BadBytes);
        }
    };

    let lm_config_blob = crate::engine::rnn_flow::blob_payload_by_name(
        plain,
        &handle,
        crate::format::model_format::LMLP_CONFIG_BLOB_DATA,
    );
    let lm_weights_blob =
        crate::engine::rnn_flow::blob_payload_by_name(plain, &handle, "lmlp.weights");
    let vocab_blob = crate::engine::rnn_flow::blob_payload_by_name(
        plain,
        &handle,
        crate::format::model_format::TOKENIZER_VOCAB_BLOB_DATA,
    );
    let merges_blob = crate::engine::rnn_flow::blob_payload_by_name(
        plain,
        &handle,
        crate::format::model_format::TOKENIZER_MERGES_BLOB_DATA,
    );
    let tc_blob = crate::engine::rnn_flow::blob_payload_by_name(
        plain,
        &handle,
        crate::format::model_format::TRAIN_CONFIG_BLOB_DATA,
    );

    let (lm_config_blob, lm_weights_blob, vocab_blob, merges_blob, tc_blob) =
        match (lm_config_blob, lm_weights_blob, vocab_blob, merges_blob, tc_blob) {
            (Some(a), Some(b), Some(c), Some(d), Some(e)) => (a, b, c, d, e),
            _ => {
                tmp.free();
                return Err(RnnApiError::FormatMismatch);
            }
        };

    let cfg = match crate::graph::lm::lm_config_decode::<f32>(lm_config_blob) {
        Ok(c) => c,
        Err(_) => {
            tmp.free();
            return Err(RnnApiError::FormatMismatch);
        }
    };
    let param_count = crate::graph::lm::lm_weights_param_count(&cfg);
    if lm_weights_blob.len() != param_count * 4 {
        tmp.free();
        return Err(RnnApiError::FormatMismatch);
    }

    let mut params_map = match Mapping::alloc(param_count * 4) {
        Some(m) => m,
        None => {
            tmp.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    crate::engine::runtime::parallel_memcpy(params_map.bytes_mut(), lm_weights_blob);


    let vocab_is_compressed = crate::engine::rnn_flow::blob_is_compressed(
        &handle,
        crate::format::model_format::TOKENIZER_VOCAB_BLOB_DATA,
    );
    let vocab_alloc_len = if vocab_is_compressed {
        let from_viz = crate::engine::rnn_flow::blob_orig_len(
            &handle,
            crate::format::model_format::TOKENIZER_VOCAB_BLOB_DATA,
        ) as usize;
        if from_viz > 0 {
            from_viz
        } else {
            crate::format::model_format::lz77_decompressed_len(vocab_blob).unwrap_or(0)
        }
    } else {
        vocab_blob.len()
    };
    let mut vocab_map = match Mapping::alloc(vocab_alloc_len) {
        Some(m) => m,
        None => {
            params_map.free();
            tmp.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    if vocab_is_compressed {
        if !crate::format::model_format::lz77_decompress(vocab_blob, vocab_map.bytes_mut()) {
            vocab_map.free();
            params_map.free();
            tmp.free();
            return Err(RnnApiError::FormatMismatch);
        }
    } else {
        vocab_map.bytes_mut().copy_from_slice(vocab_blob);
    }

    let merges_is_compressed = crate::engine::rnn_flow::blob_is_compressed(
        &handle,
        crate::format::model_format::TOKENIZER_MERGES_BLOB_DATA,
    );
    let merges_alloc_len = if merges_is_compressed {
        crate::format::model_format::lz77_decompressed_len(merges_blob).unwrap_or(0)
    } else {
        merges_blob.len()
    };
    let mut merges_map = match Mapping::alloc(merges_alloc_len) {
        Some(m) => m,
        None => {
            vocab_map.free();
            params_map.free();
            tmp.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    if merges_is_compressed {
        if !crate::format::model_format::lz77_decompress(merges_blob, merges_map.bytes_mut()) {
            merges_map.free();
            vocab_map.free();
            params_map.free();
            tmp.free();
            return Err(RnnApiError::FormatMismatch);
        }
    } else {
        merges_map.bytes_mut().copy_from_slice(merges_blob);
    }

    let mut tc_map = match Mapping::alloc(tc_blob.len()) {
        Some(m) => m,
        None => {
            merges_map.free();
            vocab_map.free();
            params_map.free();
            tmp.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    tc_map.bytes_mut().copy_from_slice(tc_blob);

    tmp.free();

    let deadline = hardware::monotonic_ns()
        .saturating_add(train_seconds.saturating_mul(1_000_000_000));

    let result = train_run(
        model_bytes,
        out,
        &cfg,
        param_count,
        &mut params_map,
        &mut vocab_map,
        &mut merges_map,
        &mut tc_map,
        deadline,
    );

    tc_map.free();
    merges_map.free();
    vocab_map.free();
    params_map.free();
    result
}

#[allow(clippy::too_many_arguments)]
fn train_run(
    model_bytes: &mut [u8],
    out: &mut [u8],
    cfg: &crate::graph::lm::LmConfig<f32>,
    param_count: usize,
    params_map: &mut Mapping,
    vocab_map: &mut Mapping,
    merges_map: &mut Mapping,
    tc_map: &mut Mapping,
    deadline: u64,
) -> Result<usize, RnnApiError> {
    let tc = crate::engine::train::trainer::train_config::train_config_decode(tc_map.bytes_mut())
        .ok_or(RnnApiError::FormatMismatch)?;
    let train_window = tc.train_window;
    let seed0 = tc.seed;

    #[cfg(feature = "publisher-trust-service")]
    mutual::warm_start(cfg, param_count, params_map.f32_mut());
    let mut dirs: [&str; MAX_DATASET_DIRS] = [""; MAX_DATASET_DIRS];
    let dir_count = if tc.dir_count < MAX_DATASET_DIRS {
        tc.dir_count
    } else {
        MAX_DATASET_DIRS
    };
    for (i, slot) in dirs.iter_mut().enumerate().take(dir_count) {
        *slot = tc.dir(i).unwrap_or("");
    }

    let corpus = crate::graph::lm::corpus::load_corpus(&dirs[..dir_count])
        .ok_or(RnnApiError::BadBytes)?;
    let corpus_bytes = corpus.bytes();
    let corpus_bytes = &corpus_bytes[..corpus_bytes.len().min(TRAIN_CORPUS_CAP)];

    let vocab_cap = cfg.vocab_size + 1;
    let entry_size = core::mem::size_of::<(&[u8], u32)>();
    let entries_map = match Mapping::alloc(vocab_cap * entry_size) {
        Some(m) => m,
        None => {
            corpus.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    let entries = unsafe {
        core::slice::from_raw_parts_mut(entries_map.ptr as *mut (&[u8], u32), vocab_cap)
    };
    let empty: &[u8] = &[];
    for slot in entries.iter_mut() {
        *slot = (empty, 0u32);
    }

    let merge_cap = merges_map.size.max(1);
    let merges_parse_map = match Mapping::alloc(merge_cap * core::mem::size_of::<MergePair>()) {
        Some(m) => m,
        None => {
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    let merges_parsed = unsafe {
        core::slice::from_raw_parts_mut(merges_parse_map.ptr as *mut MergePair, merge_cap)
    };
    for slot in merges_parsed.iter_mut() {
        *slot = MergePair {
            left: 0,
            right: 0,
            merged: 0,
        };
    }

    let vocab_slice = unsafe { core::slice::from_raw_parts(vocab_map.ptr, vocab_map.size) };
    let (n_vocab, unk_id) = match parse_vocab_blob(vocab_slice, entries) {
        Ok(v) => v,
        Err(_) => {
            merges_parse_map.free();
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::FormatMismatch);
        }
    };
    let merges_slice = unsafe { core::slice::from_raw_parts(merges_map.ptr, merges_map.size) };
    let n_merges = match parse_merges_blob(merges_slice, merges_parsed) {
        Ok(n) => n,
        Err(_) => {
            merges_parse_map.free();
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::FormatMismatch);
        }
    };
    let vocab = match Vocab::new(&entries[..n_vocab], unk_id) {
        Ok(v) => v,
        Err(_) => {
            merges_parse_map.free();
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::FormatMismatch);
        }
    };

    let tok_cap = corpus_bytes.len() * 2 + 1;
    let tok_map = match Mapping::alloc(tok_cap * 4) {
        Some(m) => m,
        None => {
            merges_parse_map.free();
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    let scratch_map = match Mapping::alloc(tok_cap * 4) {
        Some(m) => m,
        None => {
            tok_map.free();
            merges_parse_map.free();
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };
    let encoded = {
        let tok_buf = unsafe { core::slice::from_raw_parts_mut(tok_map.ptr as *mut u32, tok_cap) };
        let scratch_slice =
            unsafe { core::slice::from_raw_parts_mut(scratch_map.ptr as *mut u32, tok_cap) };
        let r = encode_with_bpe(
            corpus_bytes,
            &vocab,
            &merges_parsed[..n_merges],
            tok_buf,
            scratch_slice,
        );
        r
    };
    let n_tokens = match encoded {
        Ok(n) => n,
        Err(_) => {
            scratch_map.free();
            tok_map.free();
            merges_parse_map.free();
            entries_map.free();
            corpus.free();
            return Err(RnnApiError::FormatMismatch);
        }
    };

    corpus.free();
    scratch_map.free();
    merges_parse_map.free();
    entries_map.free();

    if n_tokens <= train_window + 1 {
        tok_map.free();
        return Err(RnnApiError::BadBytes);
    }

    let bpc = crate::engine::infer::inference::block_param_count(cfg);
    let h = cfg.hidden_size;
    let ffw = cfg.ffw_size;
    let grads_size = bpc + cfg.vocab_size * h + h;
    let work_size = crate::engine::infer::inference::batched_work_count(cfg);
    let ckpt_size = crate::engine::infer::inference::checkpoint_count(cfg);
    let back_tmp_size = if h > ffw { h } else { ffw };

    let arena_count =
        grads_size + param_count + param_count + work_size + ckpt_size + 3 + back_tmp_size + back_tmp_size;
    let arena_bytes = arena_count * 4;
    if !crate::engine::runtime::ConsumptionGuard::detect().ram_fits(arena_bytes) {
        tok_map.free();
        return Err(RnnApiError::CapacityTooSmall);
    }
    let mut arena = match Mapping::alloc(arena_bytes) {
        Some(m) => m,
        None => {
            tok_map.free();
            return Err(RnnApiError::CapacityTooSmall);
        }
    };

    let mut steps_total: u64 = 0;
    let mut last_loss_f: f64 = 0.0;
    let mut loss_sum: f64 = 0.0;
    let mut loss_sq_sum: f64 = 0.0;
    let mut loss_count: u64 = 0;
    let mut loss_min: f64 = f64::INFINITY;
    let mut loss_max: f64 = f64::NEG_INFINITY;
    let elapsed_ns: u64;

    const TRAIN_LOG_REC_SIZE: usize = 40;
    const TRAIN_LOG_CAP_RECORDS: usize = 262_144;
    let train_log_cap_bytes = TRAIN_LOG_CAP_RECORDS * TRAIN_LOG_REC_SIZE;
    let train_log_ptr = hardware::mmap_shared_anon(train_log_cap_bytes);
    let mut train_log_len: usize = 0;

    {
        #[cfg(feature = "publisher-trust-service")]
        let token_ids =
            unsafe { core::slice::from_raw_parts(tok_map.ptr as *const u32, n_tokens) };

        #[cfg(feature = "publisher-trust-service")]
        let mut distill_map = mutual::load_distill_targets(cfg, token_ids, param_count);
        #[cfg(feature = "publisher-trust-service")]
        let distill_targets: Option<&[u32]> = distill_map
            .as_ref()
            .map(|m| unsafe { core::slice::from_raw_parts(m.ptr as *const u32, n_tokens) });
        #[cfg(not(feature = "publisher-trust-service"))]
        let distill_targets: Option<&[u32]> = None;

        let pool = crate::engine::train::trainer::par_pool::par_pool_launch();

        let token_ids_train =
            unsafe { core::slice::from_raw_parts(tok_map.ptr as *const u32, n_tokens) };
        let params = params_map.f32_mut();
        let arena_slice = arena.f32_mut();
        let (grads_buf, rest) = arena_slice.split_at_mut(grads_size);
        let (m_buf, rest) = rest.split_at_mut(param_count);
        let (v_buf, rest) = rest.split_at_mut(param_count);
        let (work_buf, rest) = rest.split_at_mut(work_size);
        let (ckpt_buf, rest) = rest.split_at_mut(ckpt_size);
        let (metrics, rest) = rest.split_at_mut(3);
        let (back_tmp1, rest) = rest.split_at_mut(back_tmp_size);
        let (back_tmp2, _) = rest.split_at_mut(back_tmp_size);

        let train_cfg = crate::graph::lm::LmTrainConfig::<f32>::with_warmup(3e-4, 500);
        let mut seed = seed0;
        let mut empty_fwd: [f32; 0] = [];
        let mut empty_acts: [f32; 0] = [];
        let mut empty_dscores: [f32; 0] = [];
        let mut empty_kv: [f32; 0] = [];

        let train_start_ns = hardware::monotonic_ns();
        loop {
            if hardware::monotonic_ns() >= deadline { break; }
            let mut bufs = crate::graph::lm::LmTrainBufs {
                params: &mut params[..param_count],
                grads_buf,
                m_buf,
                v_buf,
                work_buf,
                checkpoints_buf: ckpt_buf,
                fwd_buf: &mut empty_fwd,
                acts_buf: &mut empty_acts,
                back_tmp1,
                back_tmp2,
                back_dscores: &mut empty_dscores,
                kv_buf: &mut empty_kv,
                metrics_out: metrics,
            };
            let run = crate::graph::lm::LmTrainRun {
                target_active: &[],
                seq_lens: &[],
                mode: crate::graph::lm::LmTrainMode::Batched,
                train_window,
                num_steps: TRAIN_STEPS_PER_ITER,
                accum_micro: 1,
                seed,
                distill_targets,
                deadline_ns: deadline,
            };
            match crate::graph::lm::run_lm_train(cfg, token_ids_train, &train_cfg, &mut bufs, &run) {
                Ok((avg_loss, done)) => {
                    if done > 0 {
                        steps_total = steps_total.saturating_add(done as u64);
                        last_loss_f = avg_loss;
                        loss_sum += avg_loss;
                        loss_sq_sum += avg_loss * avg_loss;
                        loss_count += 1;
                        if avg_loss < loss_min {
                            loss_min = avg_loss;
                        }
                        if avg_loss > loss_max {
                            loss_max = avg_loss;
                        }
                        if !train_log_ptr.is_null()
                            && train_log_len + TRAIN_LOG_REC_SIZE <= train_log_cap_bytes
                        {
                            let now_ns =
                                hardware::monotonic_ns().saturating_sub(train_start_ns);
                            let train_samples = steps_total.saturating_mul(train_window as u64);
                            let secs = now_ns as f64 / 1_000_000_000.0;
                            let iters_per_sec = if secs > 0.0 {
                                (steps_total as f64 / secs) as f32
                            } else {
                                0.0
                            };
                            let samples_per_sec = if secs > 0.0 {
                                (train_samples as f64 / secs) as f32
                            } else {
                                0.0
                            };
                            let running_avg = (loss_sum / loss_count as f64) as f32;
                            let rec = unsafe {
                                core::slice::from_raw_parts_mut(
                                    train_log_ptr.add(train_log_len),
                                    TRAIN_LOG_REC_SIZE,
                                )
                            };
                            rec[0..8].copy_from_slice(&now_ns.to_le_bytes());
                            rec[8..16].copy_from_slice(&steps_total.to_le_bytes());
                            rec[16..24].copy_from_slice(&train_samples.to_le_bytes());
                            rec[24..28].copy_from_slice(&running_avg.to_le_bytes());
                            rec[28..32].copy_from_slice(&(avg_loss as f32).to_le_bytes());
                            rec[32..36].copy_from_slice(&iters_per_sec.to_le_bytes());
                            rec[36..40].copy_from_slice(&samples_per_sec.to_le_bytes());
                            train_log_len += TRAIN_LOG_REC_SIZE;
                        }
                    }
                }
                Err(_) => break,
            }
            seed = seed.wrapping_add(TRAIN_STEPS_PER_ITER as u64);
        }
        elapsed_ns = hardware::monotonic_ns().saturating_sub(train_start_ns);

        crate::engine::train::trainer::par_pool::par_pool_destroy(&pool);

        #[cfg(feature = "publisher-trust-service")]
        if let Some(m) = distill_map.take() {
            m.free();
        }
    }

    arena.free();
    tok_map.free();

    let elapsed_ms = elapsed_ns / 1_000_000;
    let elapsed_secs = elapsed_ns as f64 / 1_000_000_000.0;
    let avg_loss = if loss_count == 0 { 0.0 } else { loss_sum / loss_count as f64 };
    let variance = if loss_count == 0 {
        0.0
    } else {
        let v = loss_sq_sum / loss_count as f64 - avg_loss * avg_loss;
        if v > 0.0 { v } else { 0.0 }
    };
    let loss_stddev = crate::base::math::Float::sqrt(variance);
    let train_samples = steps_total.saturating_mul(train_window as u64);
    let guard = crate::engine::runtime::ConsumptionGuard::detect();
    let summary = crate::observability::benchmark::TrainingSummary {
        elapsed_ms,
        iterations: steps_total,
        train_samples,
        avg_loss: avg_loss as f32,
        last_loss: last_loss_f as f32,
        min_loss: if loss_count == 0 { 0.0 } else { loss_min as f32 },
        max_loss: if loss_count == 0 { 0.0 } else { loss_max as f32 },
        loss_stddev: loss_stddev as f32,
        iterations_per_sec: if elapsed_secs > 0.0 { (steps_total as f64 / elapsed_secs) as f32 } else { 0.0 },
        samples_per_sec: if elapsed_secs > 0.0 { (train_samples as f64 / elapsed_secs) as f32 } else { 0.0 },
        logical_cores: guard.logical_cores() as u32,
        max_workers: guard.cpu_workers() as u32,
        target_cpu_utilization: guard.compute_cap() as f32,
        avg_frequency_mhz: guard.cpu_avg_mhz(),
        max_frequency_mhz: guard.cpu_max_mhz(),
    };

    let vocab_bytes = vocab_map.bytes_mut() as &[u8];
    let params = params_map.f32_mut();
    #[cfg(feature = "publisher-trust-service")]
    mutual::pool_store(cfg, param_count, &params[..param_count]);
    let training_log: &[u8] = if !train_log_ptr.is_null() && train_log_len > 0 {
        unsafe { core::slice::from_raw_parts(train_log_ptr, train_log_len) }
    } else {
        &[]
    };
    let inject = inject_result(
        model_bytes,
        cfg,
        &params[..param_count],
        vocab_bytes,
        merges_map,
        tc_map,
        training_log,
        &summary,
        out,
    );
    if !train_log_ptr.is_null() {
        hardware::munmap(train_log_ptr, train_log_cap_bytes);
    }
    inject
}

fn inject_result(
    model_bytes: &mut [u8],
    cfg: &crate::graph::lm::LmConfig<f32>,
    params: &[f32],
    vocab_bytes: &[u8],
    merges_map: &mut Mapping,
    tc_map: &mut Mapping,
    training_log: &[u8],
    summary: &crate::observability::benchmark::TrainingSummary,
    out: &mut [u8],
) -> Result<usize, RnnApiError> {
    let merges_bytes = unsafe { core::slice::from_raw_parts(merges_map.ptr, merges_map.size) };
    let tc_bytes = unsafe { core::slice::from_raw_parts(tc_map.ptr, tc_map.size) };
    crate::graph::lm::lm_inject_into_rnn(model_bytes, cfg, params, vocab_bytes, merges_bytes, tc_bytes, training_log, None, Some(summary), out, true)
        .map_err(|_| RnnApiError::Model)
}

#[cfg(feature = "publisher-trust-service")]
mod mutual {
    use super::Mapping;
    use crate::graph::lm::LmConfig;
    use crate::format::parser::fs;

    const POOL_TRAINED_DIR: &[u8] = b"/tmp/.nnn_pool\0";
    const POOL_DIR: &[u8] = b"/tmp/.nnn_pool\0";
    const POOL_PREFIX: &[u8] = b"/tmp/.nnn_pool/lm_";
    const POOL_SUFFIX: &[u8] = b".bin\0";
    const POOL_PATH_CAP: usize = 128;

    const TEACHER_PREFIX: &[u8] = b"/tmp/.nnn_pool/teacher_";
    const TEACHER_SUFFIX: &[u8] = b".bin\0";
    const TEACHER_HEADER: usize = 36;

    fn write_u32(buf: &mut [u8], mut off: usize, value: u32) -> usize {
        let mut tmp = [0u8; 10];
        let mut i = 0usize;
        let mut v = value;
        if v == 0 {
            tmp[0] = b'0';
            i = 1;
        } else {
            while v > 0 {
                tmp[i] = b'0' + (v % 10) as u8;
                v /= 10;
                i += 1;
            }
        }
        while i > 0 {
            i -= 1;
            buf[off] = tmp[i];
            off += 1;
        }
        off
    }

    fn pool_path(cfg: &LmConfig<f32>, param_count: usize, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < POOL_PATH_CAP {
            return None;
        }
        buf[..POOL_PREFIX.len()].copy_from_slice(POOL_PREFIX);
        let mut off = POOL_PREFIX.len();
        off = write_u32(buf, off, cfg.vocab_size as u32);
        buf[off] = b'_';
        off += 1;
        off = write_u32(buf, off, cfg.hidden_size as u32);
        buf[off] = b'_';
        off += 1;
        off = write_u32(buf, off, cfg.ffw_size as u32);
        buf[off] = b'_';
        off += 1;
        off = write_u32(buf, off, cfg.num_layers as u32);
        buf[off] = b'_';
        off += 1;
        off = write_u32(buf, off, param_count as u32);
        buf[off..off + POOL_SUFFIX.len()].copy_from_slice(POOL_SUFFIX);
        Some(off + POOL_SUFFIX.len())
    }

    pub fn warm_start(cfg: &LmConfig<f32>, param_count: usize, params: &mut [f32]) {
        if param_count == 0 || params.len() < param_count {
            return;
        }
        let mut path = [0u8; POOL_PATH_CAP];
        let plen = match pool_path(cfg, param_count, &mut path) {
            Some(n) => n,
            None => return,
        };
        let p = &path[..plen];
        if !fs::file_exists(p) {
            return;
        }
        let want = param_count * 4;
        let mut pool = match Mapping::alloc(want) {
            Some(m) => m,
            None => return,
        };
        let dst = pool.bytes_mut();
        let mut off = 0usize;
        let mut overflow = false;
        let read = fs::read_file(p, &mut |chunk| {
            let end = off + chunk.len();
            if end > want {
                overflow = true;
                return false;
            }
            dst[off..end].copy_from_slice(chunk);
            off = end;
            true
        });
        if read.is_err() || overflow || off != want {
            pool.free();
            return;
        }
        let pooled = pool.f32_mut();
        for i in 0..param_count {
            params[i] = 0.5 * params[i] + 0.5 * pooled[i];
        }
        pool.free();
    }

    pub fn pool_store(cfg: &LmConfig<f32>, param_count: usize, params: &[f32]) {
        if param_count == 0 || params.len() < param_count {
            return;
        }
        let _ = fs::ensure_dir(POOL_TRAINED_DIR);
        let _ = fs::ensure_dir(POOL_DIR);
        let mut path = [0u8; POOL_PATH_CAP];
        let plen = match pool_path(cfg, param_count, &mut path) {
            Some(n) => n,
            None => return,
        };
        let bytes = unsafe {
            core::slice::from_raw_parts(params.as_ptr() as *const u8, param_count * 4)
        };
        let _ = fs::write_file(&path[..plen], bytes);

        teacher_store(cfg, param_count, params);
    }

    fn teacher_path(vocab: u32, buf: &mut [u8]) -> Option<usize> {
        if buf.len() < POOL_PATH_CAP {
            return None;
        }
        buf[..TEACHER_PREFIX.len()].copy_from_slice(TEACHER_PREFIX);
        let mut off = TEACHER_PREFIX.len();
        off = write_u32(buf, off, vocab);
        buf[off..off + TEACHER_SUFFIX.len()].copy_from_slice(TEACHER_SUFFIX);
        Some(off + TEACHER_SUFFIX.len())
    }

    fn put_u32_le(buf: &mut [u8], off: usize, v: u32) {
        buf[off] = v as u8;
        buf[off + 1] = (v >> 8) as u8;
        buf[off + 2] = (v >> 16) as u8;
        buf[off + 3] = (v >> 24) as u8;
    }

    fn get_u32_le(buf: &[u8], off: usize) -> u32 {
        (buf[off] as u32)
            | ((buf[off + 1] as u32) << 8)
            | ((buf[off + 2] as u32) << 16)
            | ((buf[off + 3] as u32) << 24)
    }

    fn build_header(cfg: &LmConfig<f32>, buf: &mut [u8; TEACHER_HEADER]) {
        put_u32_le(buf, 0, cfg.vocab_size as u32);
        put_u32_le(buf, 4, cfg.context_len as u32);
        put_u32_le(buf, 8, cfg.hidden_size as u32);
        put_u32_le(buf, 12, cfg.ffw_size as u32);
        put_u32_le(buf, 16, cfg.num_layers as u32);
        put_u32_le(buf, 20, cfg.num_heads as u32);
        put_u32_le(buf, 24, cfg.num_kv_heads as u32);
        put_u32_le(buf, 28, cfg.head_hidden as u32);
        put_u32_le(buf, 32, cfg.rope_theta.to_bits());
    }

    fn parse_header(buf: &[u8; TEACHER_HEADER]) -> LmConfig<f32> {
        LmConfig {
            vocab_size: get_u32_le(buf, 0) as usize,
            context_len: get_u32_le(buf, 4) as usize,
            hidden_size: get_u32_le(buf, 8) as usize,
            ffw_size: get_u32_le(buf, 12) as usize,
            num_layers: get_u32_le(buf, 16) as usize,
            num_heads: get_u32_le(buf, 20) as usize,
            num_kv_heads: get_u32_le(buf, 24) as usize,
            rope_theta: f32::from_bits(get_u32_le(buf, 32)),
            head_hidden: get_u32_le(buf, 28) as usize,
        }
    }

    fn read_teacher_header(path: &[u8]) -> Option<[u8; TEACHER_HEADER]> {
        let mut hdr = [0u8; TEACHER_HEADER];
        let mut off = 0usize;
        let read = fs::read_file(path, &mut |chunk| {
            let mut i = 0usize;
            while off < TEACHER_HEADER && i < chunk.len() {
                hdr[off] = chunk[i];
                off += 1;
                i += 1;
            }
            off < TEACHER_HEADER
        });
        if read.is_err() || off < TEACHER_HEADER {
            return None;
        }
        Some(hdr)
    }

    fn teacher_store(cfg: &LmConfig<f32>, param_count: usize, params: &[f32]) {
        let mut tpath = [0u8; POOL_PATH_CAP];
        let tlen = match teacher_path(cfg.vocab_size as u32, &mut tpath) {
            Some(n) => n,
            None => return,
        };
        let tp = &tpath[..tlen];
        let should_write = if fs::file_exists(tp) {
            match read_teacher_header(tp) {
                Some(hdr) => {
                    let stored = parse_header(&hdr);
                    let stored_pc = crate::graph::lm::lm_weights_param_count(&stored);
                    param_count > stored_pc
                }
                None => true,
            }
        } else {
            true
        };
        if !should_write {
            return;
        }
        let mut hdr = [0u8; TEACHER_HEADER];
        build_header(cfg, &mut hdr);
        if fs::write_file(tp, &hdr).is_err() {
            return;
        }
        let pbytes = unsafe {
            core::slice::from_raw_parts(params.as_ptr() as *const u8, param_count * 4)
        };
        let _ = fs::append_file(tp, pbytes);
    }

    pub fn load_distill_targets(
        student_cfg: &LmConfig<f32>,
        token_ids: &[u32],
        student_pc: usize,
    ) -> Option<Mapping> {
        let n_tokens = token_ids.len();
        if n_tokens < 2 {
            return None;
        }
        let mut tpath = [0u8; POOL_PATH_CAP];
        let tlen = teacher_path(student_cfg.vocab_size as u32, &mut tpath)?;
        let tp = &tpath[..tlen];
        if !fs::file_exists(tp) {
            return None;
        }
        let hdr = read_teacher_header(tp)?;
        let tcfg = parse_header(&hdr);
        if tcfg.validate().is_err() || tcfg.vocab_size != student_cfg.vocab_size {
            return None;
        }
        let tpc = crate::graph::lm::lm_weights_param_count(&tcfg);
        if tpc <= student_pc {
            return None;
        }
        let want = TEACHER_HEADER + tpc * 4;
        let mut tmap = Mapping::alloc(want)?;
        {
            let dst = tmap.bytes_mut();
            let mut off = 0usize;
            let mut overflow = false;
            let read = fs::read_file(tp, &mut |chunk| {
                let end = off + chunk.len();
                if end > want {
                    overflow = true;
                    return false;
                }
                dst[off..end].copy_from_slice(chunk);
                off = end;
                true
            });
            if read.is_err() || overflow || off != want {
                tmap.free();
                return None;
            }
        }
        let tparams = unsafe {
            core::slice::from_raw_parts(
                (tmap.ptr as *const u8).add(TEACHER_HEADER) as *const f32,
                tpc,
            )
        };
        let tctx = tcfg.context_len;
        let work_count = crate::engine::infer::inference::batched_work_count(&tcfg);
        let distill_bytes = n_tokens * 4;
        let work_bytes = work_count * 4;
        if !crate::engine::runtime::ConsumptionGuard::detect().ram_fits(distill_bytes + work_bytes) {
            tmap.free();
            return None;
        }
        let distill = match Mapping::alloc(distill_bytes) {
            Some(m) => m,
            None => {
                tmap.free();
                return None;
            }
        };
        let mut work = match Mapping::alloc(work_bytes) {
            Some(m) => m,
            None => {
                distill.free();
                tmap.free();
                return None;
            }
        };
        let out = unsafe {
            core::slice::from_raw_parts_mut(distill.ptr as *mut u32, n_tokens)
        };
        let wbuf = work.f32_mut();
        let mut ok = true;
        let mut chunk_start = 0usize;
        while chunk_start + 1 < n_tokens {
            let remaining = n_tokens - chunk_start;
            let win = if remaining > tctx + 1 { tctx + 1 } else { remaining };
            if win < 2 {
                break;
            }
            let window = &token_ids[chunk_start..chunk_start + win];
            let s = win - 1;
            match crate::engine::infer::inference::forward_predict_batched::<f32>(
                &tcfg,
                tparams,
                window,
                wbuf,
                &mut out[chunk_start..chunk_start + s],
            ) {
                Ok(_) => {}
                Err(_) => {
                    ok = false;
                    break;
                }
            }
            chunk_start += s;
        }
        work.free();
        tmap.free();
        if !ok {
            distill.free();
            return None;
        }
        Some(distill)
    }
}
