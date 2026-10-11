use crate::math::float::Float;

pub(crate) struct CacheTensor<'a, T> {
    data: &'a mut [T],
    capacity: usize,
    head: usize,
    len: usize,
    num_heads: usize,
    head_dim: usize,
}

impl<'a, T: Float> CacheTensor<'a, T> {
    pub(crate) fn from_slice(data: &'a mut [T], capacity: usize, num_heads: usize, head_dim: usize) -> Option<Self> {
        if capacity == 0 || num_heads == 0 || head_dim == 0 { return None; }
        let required = capacity.checked_mul(num_heads)?.checked_mul(head_dim)?;
        if data.len() < required { return None; }
        Some(Self { data, capacity, head: 0, len: 0, num_heads, head_dim })
    }

    pub(crate) unsafe fn from_raw<'b>(data: *mut T, capacity: usize, num_heads: usize, head_dim: usize) -> Option<CacheTensor<'b, T>> {
        if data.is_null() || capacity == 0 || num_heads == 0 || head_dim == 0 { return None; }
        let required = capacity.checked_mul(num_heads)?.checked_mul(head_dim)?;
        let data = core::slice::from_raw_parts_mut(data, required);
        Some(CacheTensor { data, capacity, head: 0, len: 0, num_heads, head_dim })
    }

    pub(crate) fn stride(&self) -> usize {
        self.num_heads.saturating_mul(self.head_dim)
    }

    pub(crate) fn push(&mut self, kv: &[T]) -> bool {
        let s = self.stride();
        if kv.len() < s || self.head >= self.capacity { return false; }
        let off = self.head.saturating_mul(s);
        if off == usize::MAX { return false; }
        self.data[off..off + s].copy_from_slice(&kv[..s]);
        self.head = (self.head + 1) % self.capacity;
        self.len = self.len.saturating_add(1).min(self.capacity);
        true
    }

    pub(crate) fn used_tokens(&self) -> usize { self.len }

    pub(crate) fn is_full(&self) -> bool { self.len == self.capacity }

    pub(crate) fn token(&self, index: usize) -> Option<&[T]> {
        if index >= self.len { return None; }
        let first = if self.len == self.capacity { self.head } else { 0 };
        let slot = (first + index) % self.capacity;
        let offset = slot.checked_mul(self.stride())?;
        Some(&self.data[offset..offset + self.stride()])
    }

    /// Copies the live tokens, oldest first, into `out`.
    ///
    /// This replaces the old `Vec`-returning `as_slice`: the destination is
    /// caller-owned storage, so the ring is flattened without allocating. `out`
    /// must be at least `used_tokens() * stride()` long, and the byte count
    /// written is returned.
    pub(crate) fn copy_tokens_into(&self, out: &mut [T]) -> Option<usize> {
        let stride = self.stride();
        let needed = self.len.checked_mul(stride)?;

        if out.len() < needed {
            return None;
        }

        for index in 0..self.len {
            let token = self.token(index)?;
            out[index * stride..index * stride + stride].copy_from_slice(token);
        }

        Some(needed)
    }
}
