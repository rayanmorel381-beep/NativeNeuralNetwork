pub fn max_sequence_len(sequences: &[&[u32]]) -> Option<usize> {
    if sequences.is_empty() {
        return None;
    }
    let mut max_len = 0usize;
    for s in sequences {
        if s.len() > max_len {
            max_len = s.len();
        }
    }
    Some(max_len)
}

pub fn sequence_lengths(sequences: &[&[u32]], out: &mut [usize]) -> bool {
    if out.len() < sequences.len() {
        return false;
    }
    for (i, seq) in sequences.iter().enumerate() {
        out[i] = seq.len();
    }
    true
}
