#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnApiError {
    InvalidTopology,
    CapacityTooSmall,
    BadBytes,
    FormatMismatch,
    Layer,
    Model,
    NotInitialized,
}

pub(crate) fn ensure_rnn_initialized(bytes: &[u8]) -> Result<(), RnnApiError> {
    if bytes.is_empty() {
        return Err(RnnApiError::NotInitialized);
    }
    if bytes.iter().all(|b| *b == 0) {
        return Err(RnnApiError::NotInitialized);
    }
    Ok(())
}

pub(crate) fn map_engine_error(err: crate::engine::RnnFlowError) -> RnnApiError {
    match err {
        crate::engine::RnnFlowError::InvalidTopology => RnnApiError::InvalidTopology,
        crate::engine::RnnFlowError::CapacityTooSmall => RnnApiError::CapacityTooSmall,
        crate::engine::RnnFlowError::BadBytes => RnnApiError::BadBytes,
        crate::engine::RnnFlowError::Model => RnnApiError::Model,
    }
}

fn map_init_error(e: crate::base::initializers::InitError) -> RnnApiError {
    match e {
        crate::base::initializers::InitError::InvalidShape => RnnApiError::InvalidTopology,
        crate::base::initializers::InitError::ShapeMismatch => RnnApiError::CapacityTooSmall,
        crate::base::initializers::InitError::NonFinite => RnnApiError::Model,
    }
}

pub struct BuildF32Request<'a> {
    pub model_name: &'a str,
    pub topology: &'a [usize],
    pub seed: Option<u64>,
    pub weights: &'a mut [f32],
    pub biases: &'a mut [f32],
    pub runtime_input: Option<&'a [f32]>,
    pub conv_spec: Option<&'a [u8]>,
    pub conv_in_shape: [usize; 5],
    pub conv_layers: &'a [crate::graph::conv::conv_net::ConvLayerParams],
    pub benchmark_override: Option<&'a [u8]>,
    pub lm_params: &'a [usize],
    pub dataset_dirs: &'a [&'a str],
}

pub struct BuildF64Request<'a> {
    pub model_name: &'a str,
    pub topology: &'a [usize],
    pub seed: Option<u64>,
    pub weights: &'a mut [f64],
    pub biases: &'a mut [f64],
    pub runtime_input: Option<&'a [f64]>,
    pub benchmark_override: Option<&'a [u8]>,
    pub lm_params: &'a [usize],
    pub dataset_dirs: &'a [&'a str],
}

pub fn build_f32(req: BuildF32Request<'_>, buf: &mut [u8]) -> Result<usize, RnnApiError> {
    let (w_count, b_count) =
        crate::base::initializers::expected_parameter_counts(req.topology).ok_or(RnnApiError::InvalidTopology)?;
    if req.weights.len() < w_count || req.biases.len() < b_count {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if !req.conv_layers.is_empty() {
        return crate::graph::conv::conv_net::conv_build_managed(
            req.model_name,
            req.conv_in_shape,
            req.conv_layers,
            req.topology,
            req.seed.unwrap_or(0),
            &mut req.weights[..w_count],
            &mut req.biases[..b_count],
            buf,
        );
    }
    if let Some(s) = req.seed {
        crate::base::initializers::initialize_parameters_f32(
            req.topology,
            &mut req.weights[..w_count],
            &mut req.biases[..b_count],
            crate::base::initializers::InitKindF32::XavierUniform,
            s,
        )
        .map_err(map_init_error)?;
    } else {
        crate::base::initializers::initialize_parameters_f32_zeros(
            req.topology,
            &mut req.weights[..w_count],
            &mut req.biases[..b_count],
        )
        .map_err(map_init_error)?;
    }
    let precision = crate::format::model_config::Precision::F32 {
        weights: &*req.weights,
        biases: &*req.biases,
        runtime_input: req.runtime_input,
        conv_spec: req.conv_spec,
    };
    let mut benchmark_buf = [0u8; 4096];
    let benchmark = match req.benchmark_override {
        Some(b) => Some(b),
        None => {
            let n = crate::format::model_format::container::build_default_benchmark_blob(
                req.model_name,
                "f32",
                req.topology,
                0,
                0,
                0,
                &mut benchmark_buf,
            )
            .map_err(map_engine_error)?;
            Some(&benchmark_buf[..n])
        }
    };
    let dense_used = crate::format::model_format::build_with_precision(req.model_name, req.topology, &precision, benchmark, buf)?;
    if req.lm_params.is_empty() || req.dataset_dirs.is_empty() {
        return Ok(dense_used);
    }
    crate::graph::lm::embed_lm(dense_used, buf, req.lm_params, req.dataset_dirs, req.seed.unwrap_or(0), benchmark)
}

pub fn build_f64(req: BuildF64Request<'_>, buf: &mut [u8]) -> Result<usize, RnnApiError> {
    let (w_count, b_count) =
        crate::base::initializers::expected_parameter_counts(req.topology).ok_or(RnnApiError::InvalidTopology)?;
    if req.weights.len() < w_count || req.biases.len() < b_count {
        return Err(RnnApiError::CapacityTooSmall);
    }
    if let Some(s) = req.seed {
        crate::base::initializers::initialize_parameters_f64(
            req.topology,
            &mut req.weights[..w_count],
            &mut req.biases[..b_count],
            crate::base::initializers::InitKindF64::XavierUniform,
            s,
        )
        .map_err(map_init_error)?;
    } else {
        crate::base::initializers::initialize_parameters_f64_zeros(
            req.topology,
            &mut req.weights[..w_count],
            &mut req.biases[..b_count],
        )
        .map_err(map_init_error)?;
    }
    let precision = crate::format::model_config::Precision::F64 { weights: &*req.weights, biases: &*req.biases, runtime_input: req.runtime_input };
    let mut benchmark_buf = [0u8; 4096];
    let benchmark = match req.benchmark_override {
        Some(b) => Some(b),
        None => {
            let n = crate::format::model_format::container::build_default_benchmark_blob(
                req.model_name,
                "f64",
                req.topology,
                0,
                0,
                0,
                &mut benchmark_buf,
            )
            .map_err(map_engine_error)?;
            Some(&benchmark_buf[..n])
        }
    };
    let dense_used = crate::format::model_format::build_with_precision(req.model_name, req.topology, &precision, benchmark, buf)?;
    if req.lm_params.is_empty() || req.dataset_dirs.is_empty() {
        return Ok(dense_used);
    }
    crate::graph::lm::embed_lm(dense_used, buf, req.lm_params, req.dataset_dirs, req.seed.unwrap_or(0), benchmark)
}

pub fn read_rnn<'a>(
    bytes: &'a mut [u8],
    device_id: Option<&[u8]>,
) -> Result<crate::format::model_format::container::ReadableRnnFormat<'a>, RnnApiError> {
    let len = bytes.len();
    let plain_len = crate::security::crypto::decrypt_rnn(bytes, len);
    crate::format::model_format::container::read_rnn_format(&bytes[..plain_len], device_id).map_err(map_engine_error)
}

pub fn train(
    bytes: &mut [u8],
    config: &mut crate::format::model_config::ModelConfig,
) -> Result<(f64, usize), RnnApiError> {
    macro_rules! run_lm_train_branch {
        ($variant:ident) => {
            if let crate::format::model_config::Precision::$variant {
                cfg, params, token_ids, train_cfg, grads_buf, m_buf, v_buf, work_buf, checkpoints_buf,
                fwd_buf, acts_buf, back_tmp1, back_tmp2, back_dscores, kv_buf, target_active, mode,
                train_window, num_steps, accum_micro, seed, metrics_out, seq_lens,
            } = &mut config.precision
            {
                let mut bufs = crate::graph::lm::LmTrainBufs {
                    params,
                    grads_buf,
                    m_buf,
                    v_buf,
                    work_buf,
                    checkpoints_buf,
                    fwd_buf,
                    acts_buf,
                    back_tmp1,
                    back_tmp2,
                    back_dscores,
                    kv_buf,
                    metrics_out,
                };
                let run = crate::graph::lm::LmTrainRun {
                    target_active,
                    seq_lens,
                    mode: *mode,
                    train_window: *train_window,
                    num_steps: *num_steps,
                    accum_micro: *accum_micro,
                    seed: *seed,
                    distill_targets: None,
                    deadline_ns: 0,
                };
                return crate::graph::lm::run_lm_train(cfg, token_ids, train_cfg, &mut bufs, &run);
            }
        };
    }

    run_lm_train_branch!(LmTrainF32);
    run_lm_train_branch!(LmTrainF64);

    ensure_rnn_initialized(bytes)?;
    if crate::engine::rnn_flow::blob_range(bytes, crate::format::model_format::BLOB_CONV_SPEC).is_ok() {
        return crate::graph::conv::conv_net::conv_train_managed(bytes, config);
    }
    crate::engine::train::trainer::train(bytes, config).map_err(map_engine_error)
}

pub fn run(
    bytes: &mut [u8],
    config: &mut crate::format::model_config::ModelConfig,
) -> Result<(f64, usize), RnnApiError> {
    crate::engine::infer::inference::run_inference(bytes, config)
}
