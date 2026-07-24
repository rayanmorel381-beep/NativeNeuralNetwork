use crate::format::model_format::{
    encoded_size, header::is_header_consistent, parse_header, scalar_bytes_for_dtype,
};

pub fn expected_encoded_size_from_header(bytes: &[u8]) -> Option<usize> {
    let h = parse_header(bytes)?;
    if !is_header_consistent(&h) {
        return None;
    }
    let scalar_bytes = scalar_bytes_for_dtype(h.dtype)?;
    encoded_size(scalar_bytes, h.layer_count, h.weights_len, h.biases_len)
}

pub fn has_full_payload(bytes: &[u8]) -> bool {
    let expected = match expected_encoded_size_from_header(bytes) {
        Some(v) => v,
        None => return false,
    };
    bytes.len() == expected
}
