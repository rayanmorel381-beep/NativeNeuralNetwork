use crate::math::float::Float;

pub(crate) struct TrainingTensor<'a, T> {
    data: &'a mut [T],
    grad: &'a mut [T],
    shape: [usize; 5],
}

impl<'a, T: Float> TrainingTensor<'a, T> {
    pub(crate) fn from_slices(data: &'a mut [T], grad: &'a mut [T], shape: [usize; 5]) -> Option<Self> {
        let numel = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        })?;
        if data.len() != numel || grad.len() != numel {
            return None;
        }
        Some(Self { data, grad, shape })
    }

    pub(crate) fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() { *g = T::ZERO; }
    }

    pub(crate) fn data(&self) -> &[T] { self.data }
    pub(crate) fn data_mut(&mut self) -> &mut [T] { self.data }
    pub(crate) fn gradients(&self) -> &[T] { self.grad }
    pub(crate) fn shape(&self) -> [usize; 5] { self.shape }

    pub(crate) fn accumulate_grad(&mut self, delta: &[T]) {
        let n = self.grad.len().min(delta.len());
        for (i, grad) in self.grad.iter_mut().enumerate().take(n) {
            *grad += delta[i];
        }
    }

    pub(crate) fn apply_grad(&mut self, lr: T) {
        let n = self.data.len().min(self.grad.len());
        for (i, data) in self.data.iter_mut().enumerate().take(n) {
            *data -= self.grad[i] * lr;
        }
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        }).unwrap_or(0)
    }
}
