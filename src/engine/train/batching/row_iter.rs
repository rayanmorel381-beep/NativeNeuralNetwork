pub fn for_each_token_row(
    tokens: &[u32],
    row_width: usize,
    mut f: impl FnMut(usize, &[u32]),
) -> bool {
    if row_width == 0 || !tokens.len().is_multiple_of(row_width) {
        return false;
    }
    let rows = tokens.len() / row_width;
    for r in 0..rows {
        let off = r * row_width;
        f(r, &tokens[off..off + row_width]);
    }
    true
}

pub fn count_non_pad(tokens: &[u32], pad_id: u32) -> usize {
    let mut c = 0usize;
    for &t in tokens {
        if t != pad_id {
            c += 1;
        }
    }
    c
}
