pub fn train(
    model_bytes: &mut [u8],
    train_seconds: u64,
    out: &mut [u8],
) -> Result<usize, super::core_api::RnnApiError> {
    crate::engine::train::trainer::train_lm(model_bytes, train_seconds, out)
}
