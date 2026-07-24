use crate::security::crypto::constant_time_eq;
use core::convert::TryInto;

pub const RMD1_HEADER_SIZE: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Header {
    pub version: u16,
    pub dtype: u8,
    pub layer_count: usize,
    pub weights_len: usize,
    pub biases_len: usize,
}

pub fn write_header(
    out: &mut [u8],
    dtype: u8,
    layer_count: usize,
    weights_len: usize,
    biases_len: usize,
) -> Option<()> {
    if out.len() < RMD1_HEADER_SIZE {
        return None;
    }
    if dtype > 1 {
        return None;
    }

    let layer_count_u32 = u32::try_from(layer_count).ok()?;
    let weights_len_u32 = u32::try_from(weights_len).ok()?;
    let biases_len_u32 = u32::try_from(biases_len).ok()?;

    out[0..4].copy_from_slice(b"RMD1");
    out[4..6].copy_from_slice(&1u16.to_le_bytes());
    out[6] = dtype;
    out[7] = 0u8;
    out[8..12].copy_from_slice(&layer_count_u32.to_le_bytes());
    out[12..16].copy_from_slice(&weights_len_u32.to_le_bytes());
    out[16..20].copy_from_slice(&biases_len_u32.to_le_bytes());
    Some(())
}

pub fn parse_header(bytes: &[u8]) -> Option<Header> {
    if bytes.len() < RMD1_HEADER_SIZE {
        return None;
    }
    if !constant_time_eq(&bytes[0..4], b"RMD1") {
        return None;
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().ok()?);
    let dtype = bytes[6];
    let layer_count = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let weights_len = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    let biases_len = u32::from_le_bytes(bytes[16..20].try_into().ok()?) as usize;
    Some(Header {
        version,
        dtype,
        layer_count,
        weights_len,
        biases_len,
    })
}

pub fn is_header_consistent(h: &Header) -> bool {
    if h.version != 1 {
        return false;
    }
    if h.dtype > 1 {
        return false;
    }
    h.layer_count > 0
}
