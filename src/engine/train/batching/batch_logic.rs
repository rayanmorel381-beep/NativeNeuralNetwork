#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchError {
    ShapeMismatch,
    Empty,
}

pub fn pad_sequences_u32(
    sequences: &[&[u32]],
    pad_id: u32,
    out: &mut [u32],
    max_len: usize,
) -> Result<(), BatchError> {
    if sequences.is_empty() || max_len == 0 {
        return Err(BatchError::Empty);
    }
    let needed = sequences
        .len()
        .checked_mul(max_len)
        .ok_or(BatchError::ShapeMismatch)?;
    if out.len() < needed {
        return Err(BatchError::ShapeMismatch);
    }

    for (b, seq) in sequences.iter().enumerate() {
        let row = &mut out[b * max_len..(b + 1) * max_len];
        for (t, cell) in row.iter_mut().enumerate().take(max_len) {
            *cell = seq.get(t).copied().unwrap_or(pad_id);
        }
    }
    Ok(())
}

pub fn make_padding_mask(
    tokens: &[u32],
    pad_id: u32,
    out_mask: &mut [u8],
) -> Result<(), BatchError> {
    if tokens.is_empty() {
        return Err(BatchError::Empty);
    }
    if out_mask.len() < tokens.len() {
        return Err(BatchError::ShapeMismatch);
    }

    for (i, &tok) in tokens.iter().enumerate() {
        out_mask[i] = if tok == pad_id { 0 } else { 1 };
    }
    Ok(())
}
