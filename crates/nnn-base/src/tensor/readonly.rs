use crate::math::float::Float;

pub(crate) struct TensorViewRo<'a, T: Float> {
    pub(crate) data: &'a [T],
    pub(crate) shape: [usize; 5],
}

impl<'a, T: Float> TensorViewRo<'a, T> {
    pub(crate) fn from_slice(data: &'a [T], shape: [usize; 5]) -> Self {
        Self { data, shape }
    }

    pub(crate) fn len(&self) -> usize {
        self.shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        }).unwrap_or(0)
    }

    pub(crate) fn is_empty(&self) -> bool { self.len() == 0 }

    pub(crate) fn is_valid_layout(&self) -> bool { self.len() == self.data.len() }

    pub(crate) fn checksum(&self) -> u64 {
        let mut h: u64 = 14695981039346656037u64;
        let bytes = unsafe { core::slice::from_raw_parts(
            self.data.as_ptr() as *const u8,
            core::mem::size_of_val(self.data),
        ) };
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211u64);
        }
        h
    }
}
