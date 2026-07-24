use crate::base::math::Float;

pub struct CacheTensor<T> {
    data: *mut T,
    capacity: usize,
    head: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

unsafe impl<T: Send> Send for CacheTensor<T> {}
unsafe impl<T: Sync> Sync for CacheTensor<T> {}

impl<T: Float> CacheTensor<T> {
    pub fn from_raw(data: *mut T, capacity: usize, num_heads: usize, head_dim: usize) -> Option<Self> {
        if data.is_null() || capacity == 0 || num_heads == 0 || head_dim == 0 { return None; }
        Some(Self { data, capacity, head: 0, num_heads, head_dim })
    }

    pub fn stride(&self) -> usize {
        self.num_heads.saturating_mul(self.head_dim)
    }

    pub fn push(&mut self, kv: &[T]) -> bool {
        let s = self.stride();
        if kv.len() < s || self.head >= self.capacity { return false; }
        let off = self.head * s;
        unsafe {
            core::ptr::copy_nonoverlapping(kv.as_ptr(), self.data.add(off), s);
        }
        self.head = (self.head + 1) % self.capacity;
        true
    }

    pub fn used_tokens(&self) -> usize { self.head }

    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.data, self.capacity * self.stride()) }
    }
}
