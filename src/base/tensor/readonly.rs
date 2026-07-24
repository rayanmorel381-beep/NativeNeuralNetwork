pub struct TensorViewRo<'a> {
    pub data: &'a [f32],
    pub shape: [usize; 5],
}

impl<'a> TensorViewRo<'a> {
    pub fn from_slice(data: &'a [f32], shape: [usize; 5]) -> Self {
        Self { data, shape }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn checksum(&self) -> u64 {
        let mut h: u64 = 14695981039346656037u64;
        let bytes = unsafe {
            core::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * 4,
            )
        };
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211u64);
        }
        h
    }
}
