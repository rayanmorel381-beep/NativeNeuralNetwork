use crate::base::math::Float;

pub struct StaticTensor<T, const N: usize> {
    pub data: [T; N],
    pub shape: [usize; 5],
    pub used: usize,
}

impl<T: Float, const N: usize> StaticTensor<T, N> {
    pub fn zeroed(shape: [usize; 5]) -> Self {
        let mut data: core::mem::MaybeUninit<[T; N]> = core::mem::MaybeUninit::uninit();
        let ptr = data.as_mut_ptr() as *mut T;
        for i in 0..N {
            unsafe { ptr.add(i).write(T::ZERO); }
        }
        Self { data: unsafe { data.assume_init() }, shape, used: N }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.used]
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.used]
    }

    pub fn fill_zero(&mut self) {
        for v in self.data.iter_mut() { *v = T::ZERO; }
    }
}
