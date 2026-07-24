use crate::base::math::Float;

pub struct TensorViewMut<'a, T> {
    pub data: &'a mut [T],
    pub shape: [usize; 5],
    pub offset: usize,
}

impl<'a, T: Float> TensorViewMut<'a, T> {
    pub fn from_slice(data: &'a mut [T], shape: [usize; 5]) -> Self {
        Self { data, shape, offset: 0 }
    }

    pub fn from_slice_offset(data: &'a mut [T], shape: [usize; 5], offset: usize) -> Self {
        Self { data, shape, offset }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn fill(&mut self, val: T) {
        for v in self.data.iter_mut() { *v = val; }
    }

    pub fn axpy(&mut self, alpha: T, src: &[T]) {
        let n = self.data.len().min(src.len());
        for i in 0..n {
            self.data[i] += alpha * src[i];
        }
    }
}
