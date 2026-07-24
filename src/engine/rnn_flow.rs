use crate::format::model_format::RnnProtocolError;
use crate::format::rnn_format::lookup::find_blob_index;
use crate::format::rnn_format::parser::parse_rnn_from_bytes;
use crate::base::scratch::Scratch;

pub const MAX_LAYERS: usize = 128;
pub const LAYER_META_ENTRY_SIZE: usize = 20;
pub(crate) const MAX_TOTAL_NEURONS: usize = 65536;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RnnFlowError {
    InvalidTopology,
    CapacityTooSmall,
    BadBytes,
    Model,
}

pub(crate) fn blob_range(
    bytes: &[u8],
    name: &str,
) -> Result<(usize, usize), RnnFlowError> {
    let mut scratch_buf = [0u8; 16384];
    let mut scratch = Scratch::new(&mut scratch_buf);
    let handle = parse_rnn_from_bytes(bytes, &mut scratch)
        .map_err(|_| RnnFlowError::BadBytes)?;
    let idx = find_blob_index(&handle, name).ok_or(RnnFlowError::Model)?;
    let desc = handle.blobs.get(idx).ok_or(RnnFlowError::Model)?;
    let start = usize::try_from(desc.offset).map_err(|_| RnnFlowError::BadBytes)?;
    let len = usize::try_from(desc.length).map_err(|_| RnnFlowError::BadBytes)?;
    let end = start.checked_add(len).ok_or(RnnFlowError::BadBytes)?;
    if end > bytes.len() {
        return Err(RnnFlowError::BadBytes);
    }
    Ok((start, end))
}

pub fn rnn_dtype(bytes: &[u8]) -> Result<u8, RnnFlowError> {
    let request = crate::format::model_config::rnn::ValidateRequest::RnnDtype { bytes };
    match crate::format::model_config::rnn::validate(request) {
        Ok(crate::format::model_config::rnn::ValidateResult::RnnDtype(dtype)) => Ok(dtype),
        Ok(_) => Err(RnnFlowError::BadBytes),
        Err(_) => Err(RnnFlowError::BadBytes),
    }
}

pub(crate) fn bytes_as_f32_mut(b: &mut [u8]) -> &mut [f32] {
    let count = b.len() / 4;
    assert!(b.len().is_multiple_of(4));
    let ptr = b.as_mut_ptr();
    assert!((ptr as usize).is_multiple_of(core::mem::align_of::<f32>()));
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut f32, count) }
}

pub(crate) fn bytes_as_f64_mut(b: &mut [u8]) -> &mut [f64] {
    let count = b.len() / 8;
    assert!(b.len().is_multiple_of(8));
    let ptr = b.as_mut_ptr();
    assert!((ptr as usize).is_multiple_of(core::mem::align_of::<f64>()));
    unsafe { core::slice::from_raw_parts_mut(ptr as *mut f64, count) }
}

pub(crate) fn blob_payload_by_name<'a>(
    bytes: &'a [u8],
    handle: &crate::format::rnn_format::parser::RnnHandle<'_, '_>,
    name: &str,
) -> Option<&'a [u8]> {
    let idx = find_blob_index(handle, name)?;
    let desc = handle.blobs.get(idx)?;
    let start = usize::try_from(desc.offset).ok()?;
    let len = usize::try_from(desc.length).ok()?;
    let end = start.checked_add(len)?;
    if end > bytes.len() {
        return None;
    }
    Some(&bytes[start..end])
}

pub(crate) fn blob_is_compressed(
    handle: &crate::format::rnn_format::parser::RnnHandle<'_, '_>,
    name: &str,
) -> bool {
    if let Some(idx) = find_blob_index(handle, name) {
        if let Some(desc) = handle.blobs.get(idx) {
            return desc.is_compressed;
        }
    }
    false
}

pub(crate) fn blob_orig_len(
    handle: &crate::format::rnn_format::parser::RnnHandle<'_, '_>,
    name: &str,
) -> u64 {
    if let Some(idx) = find_blob_index(handle, name) {
        if let Some(desc) = handle.blobs.get(idx) {
            return desc.orig_length;
        }
    }
    0
}

pub(crate) fn map_protocol_error(err: RnnProtocolError) -> RnnFlowError {
    match err {
        RnnProtocolError::CapacityTooSmall => RnnFlowError::CapacityTooSmall,
        _ => RnnFlowError::BadBytes,
    }
}
