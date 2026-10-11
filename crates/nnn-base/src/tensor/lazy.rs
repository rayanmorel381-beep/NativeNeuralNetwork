use crate::math::float::Float;

pub(crate) struct LazyTensor<T> {
    eval_fn: fn(*const u8, &mut [T]),
    opaque: *const u8,
    pub(crate) shape: [usize; 5],
}

unsafe impl<T: Send> Send for LazyTensor<T> {}
unsafe impl<T: Sync> Sync for LazyTensor<T> {}

impl<T: Float> LazyTensor<T> {
    pub(crate) fn new(eval_fn: fn(*const u8, &mut [T]), opaque: *const u8, shape: [usize; 5]) -> Self {
        Self { eval_fn, opaque, shape }
    }

    pub(crate) fn evaluate(&self, out: &mut [T]) {
        (self.eval_fn)(self.opaque, out);
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
    }
}
