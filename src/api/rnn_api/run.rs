pub fn run(
    bytes: &mut [u8],
    config: &mut crate::format::model_config::ModelConfig,
) -> Result<(f64, usize), super::core_api::RnnApiError> {
    super::core_api::run(bytes, config)
}
