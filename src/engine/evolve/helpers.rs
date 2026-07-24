use crate::engine::rnn_flow::{blob_range, RnnFlowError, MAX_LAYERS};
use crate::format::model_format::{header_tlv_payload, BLOB_LAYER_META};

pub(super) const LAYER_META_ENTRY_SIZE: usize = 20;

pub(super) struct EvolveCtx<'a> {
    pub(super) model_name: &'a str,
    pub(super) old_topo: &'a [usize],
    pub(super) new_topo: &'a [usize],
    pub(super) old_w_bytes: &'a [u8],
    pub(super) old_b_bytes: &'a [u8],
    pub(super) seed: u64,
}

pub(super) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    pub(super) fn next_f32(&mut self) -> f32 {
        let raw = (self.next_u64() >> 40) as u32;
        raw as f32 / 16777216.0
    }

    pub(super) fn next_f64(&mut self) -> f64 {
        let raw = self.next_u64() >> 11;
        raw as f64 / 9007199254740992.0f64
    }
}

pub(super) fn xavier_limit_f32(fan_in: usize, fan_out: usize) -> f32 {
    let s = 6.0f32 / (fan_in as f32 + fan_out as f32);
    crate::base::math::sqrtf(s)
}

pub(super) fn xavier_limit_f64(fan_in: usize, fan_out: usize) -> f64 {
    let s = 6.0f64 / (fan_in as f64 + fan_out as f64);
    crate::base::math::sqrtd(s)
}

pub(super) fn parse_topology(bytes: &[u8]) -> Result<([usize; MAX_LAYERS], usize), RnnFlowError> {
    let (lm_start, lm_end) = blob_range(bytes, BLOB_LAYER_META)?;
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
    for idx in 0..layer_count {
        let base = lm_start + idx * LAYER_META_ENTRY_SIZE;
        let input_size = u32::from_le_bytes([
            bytes[base], bytes[base + 1], bytes[base + 2], bytes[base + 3],
        ]) as usize;
        let output_size = u32::from_le_bytes([
            bytes[base + 4], bytes[base + 5], bytes[base + 6], bytes[base + 7],
        ]) as usize;
        if idx == 0 {
            topology[0] = input_size;
            topo_len = 1;
        }
        topology[topo_len] = output_size;
        topo_len += 1;
    }
    if topo_len < 2 {
        return Err(RnnFlowError::BadBytes);
    }
    Ok((topology, topo_len))
}

pub(super) fn extract_model_name(bytes: &[u8]) -> &str {
    header_tlv_payload(bytes, crate::format::model_config::ingest::TLV_HEADER_MODEL_NAME)
        .ok()
        .and_then(|v| core::str::from_utf8(v).ok())
        .unwrap_or("")
}

pub(super) fn bytes_as_f32(b: &[u8]) -> &[f32] {
    let count = b.len() / 4;
    let ptr = b.as_ptr();
    unsafe { core::slice::from_raw_parts(ptr as *const f32, count) }
}

pub(super) fn bytes_as_f64(b: &[u8]) -> &[f64] {
    let count = b.len() / 8;
    let ptr = b.as_ptr();
    unsafe { core::slice::from_raw_parts(ptr as *const f64, count) }
}

pub(super) fn write_f32(dst: &mut [u8], offset: usize, val: f32) {
    dst[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

pub(super) fn write_f64(dst: &mut [u8], offset: usize, val: f64) {
    dst[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
}

pub(super) fn read_f32(src: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([src[offset], src[offset + 1], src[offset + 2], src[offset + 3]])
}

pub(super) fn read_f64(src: &[u8], offset: usize) -> f64 {
    f64::from_le_bytes([
        src[offset], src[offset + 1], src[offset + 2], src[offset + 3],
        src[offset + 4], src[offset + 5], src[offset + 6], src[offset + 7],
    ])
}
