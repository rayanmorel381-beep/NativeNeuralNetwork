pub fn read_rnn<'a>(
    bytes: &'a mut [u8],
    device_id: Option<&[u8]>,
) -> Result<crate::format::model_format::container::ReadableRnnFormat<'a>, super::core_api::RnnApiError> {
    super::core_api::read_rnn(bytes, device_id)
}
