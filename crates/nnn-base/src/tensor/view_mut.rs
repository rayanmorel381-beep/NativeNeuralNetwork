use crate::math::float::Float;

pub(crate) struct TensorViewMut<'a, T> {
    pub(crate) data: &'a mut [T],
    pub(crate) shape: [usize; 5],
    pub(crate) offset: usize,
}

impl<'a, T: Float> TensorViewMut<'a, T> {
    pub(crate) fn from_slice(data: &'a mut [T], shape: [usize; 5]) -> Option<Self> {
        Self::from_slice_offset(data, shape, 0)
    }

    pub(crate) fn from_slice_offset(data: &'a mut [T], shape: [usize; 5], offset: usize) -> Option<Self> {
        let len = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        })?;
        let end = offset.checked_add(len)?;
        if end > data.len() { return None; }
        Some(Self { data, shape, offset })
    }

    pub(crate) fn len(&self) -> usize {
        self.shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        }).unwrap_or(0)
    }

    pub(crate) fn is_empty(&self) -> bool { self.len() == 0 }

    pub(crate) fn fill(&mut self, val: T) {
        let end = self.offset.saturating_add(self.len()).min(self.data.len());
        for v in self.data[self.offset..end].iter_mut() { *v = val; }
    }

    pub(crate) fn axpy(&mut self, alpha: T, src: &[T]) {
        let end = self.offset.saturating_add(self.len()).min(self.data.len());
        let n = end.saturating_sub(self.offset).min(src.len());
        for (i, data) in self.data[self.offset..end].iter_mut().enumerate().take(n) {
            *data += alpha * src[i];
        }
    }
}
