pub trait ConstantTimeEq {
    fn ct_eq(&self, other: &Self) -> bool;
}

impl ConstantTimeEq for [u8] {
    fn ct_eq(&self, other: &Self) -> bool {
        constant_time_eq(self, other)
    }
}

impl<const N: usize> ConstantTimeEq for [u8; N] {
    fn ct_eq(&self, other: &Self) -> bool {
        constant_time_eq(self.as_ref(), other.as_ref())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DigestToHexError {
    BufferTooSmall,
}

impl core::fmt::Display for DigestToHexError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DigestToHexError::BufferTooSmall => write!(
                f,
                "buffer too small for hex digest (need 2 * digest length)"
            ),
        }
    }
}

pub fn digest_to_hex_lower<const N: usize>(
    digest: &[u8; N],
    out: &mut [u8],
) -> Result<usize, DigestToHexError> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let needed = N * 2;
    if out.len() < needed {
        return Err(DigestToHexError::BufferTooSmall);
    }
    for i in 0..N {
        let b = digest[i];
        out[i * 2] = HEX[(b >> 4) as usize];
        out[i * 2 + 1] = HEX[(b & 0x0f) as usize];
    }
    Ok(needed)
}

pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff_len = a.len() ^ b.len();
    let mut diff = 0u8;
    while diff_len > 0 {
        diff |= (diff_len & 0xff) as u8;
        diff_len >>= 8;
    }

    let max_len = core::cmp::max(a.len(), b.len());
    for i in 0..max_len {
        let ai = if i < a.len() { a[i] } else { 0 };
        let bi = if i < b.len() { b[i] } else { 0 };
        diff |= ai ^ bi;
    }

    diff == 0
}
