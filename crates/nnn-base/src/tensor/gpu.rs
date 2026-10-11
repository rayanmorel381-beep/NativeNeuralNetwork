pub(crate) struct GpuTensor {
    ptr: *mut u8,
    byte_len: usize,
    shape: [usize; 5],
}

unsafe impl Send for GpuTensor {}
unsafe impl Sync for GpuTensor {}

impl GpuTensor {
    pub(crate) fn from_raw(ptr: *mut u8, byte_len: usize, shape: [usize; 5]) -> Option<Self> {
        if ptr.is_null() || byte_len == 0 { return None; }
        let elements = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        })?;
        if elements.checked_mul(core::mem::size_of::<f32>())? > byte_len {
            return None;
        }
        Some(Self { ptr, byte_len, shape })
    }

    pub(crate) fn as_bytes(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.byte_len) }
    }

    pub(crate) fn byte_len(&self) -> usize { self.byte_len }
    pub(crate) fn shape(&self) -> [usize; 5] { self.shape }

    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.byte_len) }
    }

    pub(crate) fn numel_f32(&self) -> usize {
        self.byte_len / core::mem::size_of::<f32>()
    }
}
