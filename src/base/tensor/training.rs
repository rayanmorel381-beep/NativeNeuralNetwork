use crate::base::math::Float;

pub struct TrainingTensor<'a, T> {
    pub data: &'a mut [T],
    pub grad: &'a mut [T],
    pub shape: [usize; 5],
}

impl<'a, T: Float> TrainingTensor<'a, T> {
    pub fn from_slices(data: &'a mut [T], grad: &'a mut [T], shape: [usize; 5]) -> Self {
        Self { data, grad, shape }
    }

    pub fn zero_grad(&mut self) {
        for g in self.grad.iter_mut() { *g = T::ZERO; }
    }

    pub fn accumulate_grad(&mut self, delta: &[T]) {
        let n = self.grad.len().min(delta.len());
        for i in 0..n {
            self.grad[i] += delta[i];
        }
    }

    pub fn apply_grad(&mut self, lr: T) {
        let n = self.data.len().min(self.grad.len());
        for i in 0..n {
            self.data[i] -= self.grad[i] * lr;
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
    }
}
