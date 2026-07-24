pub struct GpuTensor {
    pub ptr: *mut u8,
    pub byte_len: usize,
    pub shape: [usize; 5],
}

unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

impl GpuTensor {
    pub fn from_raw(ptr: *mut u8, byte_len: usize, shape: [usize; 5]) -> Option<Self> {
        if ptr.is_null() || byte_len == 0 { return None; }
        Some(Self { ptr, byte_len, shape })
    }

    pub fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.byte_len) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.byte_len) }
    }

    pub fn numel_f32(&self) -> usize {
        self.byte_len / core::mem::size_of::<f32>()
    }
}
