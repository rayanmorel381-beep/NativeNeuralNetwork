use crate::math::float::Float;

pub(crate) struct StaticTensor<T, const N: usize> {
    pub(crate) data: [T; N],
    pub(crate) shape: [usize; 5],
    pub(crate) used: usize,
}

impl<T: Float, const N: usize> StaticTensor<T, N> {
    pub(crate) fn zeroed(shape: [usize; 5]) -> Option<Self> {
        let used = shape.iter().try_fold(1usize, |count, &dimension| count.checked_mul(dimension))?;
        if used != N { return None; }
        let mut data: core::mem::MaybeUninit<[T; N]> = core::mem::MaybeUninit::uninit();
        let ptr = data.as_mut_ptr() as *mut T;
        for i in 0..N {
            unsafe { ptr.add(i).write(T::ZERO); }
        }
        Some(Self { data: unsafe { data.assume_init() }, shape, used })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data[..self.used]
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.used]
    }

    pub(crate) fn fill_zero(&mut self) {
        for v in self.data.iter_mut() { *v = T::ZERO; }
    }
}
