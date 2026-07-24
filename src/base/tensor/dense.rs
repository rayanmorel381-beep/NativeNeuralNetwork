use crate::engine::runtime::hardware;
use crate::base::math::Float;

pub struct DenseTensor<T: Float> {
    ptr: *mut T,
    len: usize,
    pub shape: [usize; 5],
}

unsafe impl<T: Float + Send> Send for DenseTensor<T> {}
unsafe impl<T: Float + Sync> Sync for DenseTensor<T> {}

impl<T: Float> DenseTensor<T> {
    pub fn alloc(shape: [usize; 5]) -> Option<Self> {
        let len = shape.iter().fold(1usize, |a, &d| a.saturating_mul(d));
        if len == 0 { return None; }
        let byte_size = len.checked_mul(T::BYTE_SIZE)?;
        let ptr = hardware::mmap_private_anon(byte_size) as *mut T;
        if ptr.is_null() { return None; }
        Some(Self { ptr, len, shape })
    }

    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize { self.len }

    pub fn is_empty(&self) -> bool { self.len == 0 }
}

impl<T: Float> Drop for DenseTensor<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            hardware::munmap(self.ptr as *mut u8, self.len * T::BYTE_SIZE);
        }
    }
}
