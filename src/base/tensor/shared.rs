use core::sync::atomic::{AtomicUsize, Ordering};
use crate::engine::runtime::hardware;

struct SharedInner {
    refcount: AtomicUsize,
    byte_len: usize,
}

pub struct SharedTensor {
    inner: *mut SharedInner,
    data: *mut f32,
    len: usize,
    pub shape: [usize; 5],
}

unsafe impl Send for SharedTensor {}
unsafe impl Sync for SharedTensor {}

impl SharedTensor {
    pub fn alloc(shape: [usize; 5]) -> Option<Self> {
        let len = shape.iter().fold(1usize, |a, &d| a.saturating_mul(d));
        if len == 0 { return None; }
        let data_bytes = len.checked_mul(core::mem::size_of::<f32>())?;
        let header_bytes = core::mem::size_of::<SharedInner>();
        let total = header_bytes.checked_add(data_bytes)?;
        let raw = hardware::mmap_private_anon(total);
        if raw.is_null() { return None; }
        let inner = raw as *mut SharedInner;
        unsafe {
            (*inner).refcount = AtomicUsize::new(1);
            (*inner).byte_len = data_bytes;
        }
        let data = unsafe { raw.add(header_bytes) } as *mut f32;
        Some(Self { inner, data, len, shape })
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { core::slice::from_raw_parts(self.data, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { core::slice::from_raw_parts_mut(self.data, self.len) }
    }

    pub fn ref_count(&self) -> usize {
        unsafe { (*self.inner).refcount.load(Ordering::Acquire) }
    }
}

impl Clone for SharedTensor {
    fn clone(&self) -> Self {
        unsafe { (*self.inner).refcount.fetch_add(1, Ordering::AcqRel) };
        Self { inner: self.inner, data: self.data, len: self.len, shape: self.shape }
    }
}

impl Drop for SharedTensor {
    fn drop(&mut self) {
        let prev = unsafe { (*self.inner).refcount.fetch_sub(1, Ordering::AcqRel) };
        if prev == 1 {
            let data_bytes = unsafe { (*self.inner).byte_len };
            let header_bytes = core::mem::size_of::<SharedInner>();
            hardware::munmap(self.inner as *mut u8, header_bytes + data_bytes);
        }
    }
}
