pub fn build_f32_with_request(
    req: super::core_api::BuildF32Request<'_>,
    buf: &mut [u8],
) -> Result<usize, super::core_api::RnnApiError> {
    super::core_api::build_f32(req, buf)
}

#[allow(clippy::too_many_arguments)]
pub fn build_f32(
    model_name: &str,
    topology: &[usize],
    seed: impl Into<Option<u64>>,
    weights: &mut [f32],
    biases: &mut [f32],
    lm_params: &[usize],
    dataset_dirs: &[&str],
    buf: &mut [u8],
) -> Result<usize, super::core_api::RnnApiError> {
    let req = super::core_api::BuildF32Request {
        model_name,
        topology,
        seed: seed.into(),
        weights,
        biases,
        runtime_input: None,
        conv_spec: None,
        conv_in_shape: [0; 5],
        conv_layers: &[],
        benchmark_override: None,
        lm_params,
        dataset_dirs,
    };
    build_f32_with_request(req, buf)
}

pub fn build_f64_with_request(
    req: super::core_api::BuildF64Request<'_>,
    buf: &mut [u8],
) -> Result<usize, super::core_api::RnnApiError> {
    super::core_api::build_f64(req, buf)
}

#[allow(clippy::too_many_arguments)]
pub fn build_f64(
    model_name: &str,
    topology: &[usize],
    seed: impl Into<Option<u64>>,
    weights: &mut [f64],
    biases: &mut [f64],
    lm_params: &[usize],
    dataset_dirs: &[&str],
    buf: &mut [u8],
) -> Result<usize, super::core_api::RnnApiError> {
    let req = super::core_api::BuildF64Request {
        model_name,
        topology,
        seed: seed.into(),
        weights,
        biases,
        runtime_input: None,
        benchmark_override: None,
        lm_params,
        dataset_dirs,
    };
    build_f64_with_request(req, buf)
}

