use crate::math::float::Float;

/// A dense tensor borrowed over caller-owned storage.
///
/// The crate never owns tensor storage: every buffer comes from `scratch`, so a
/// dense tensor is a shape laid over a slice rather than an allocation it has to
/// free. That keeps the type `Copy`-free but `Drop`-free, and it removes the
/// last reason a tensor would need a global allocator.
pub(crate) struct DenseTensor<'a, T> {
    data: &'a mut [T],
    shape: [usize; 5],
}

impl<'a, T: Float> DenseTensor<'a, T> {
    /// Binds `shape` to `data`, which must hold exactly the shape's product.
    pub(crate) fn from_slice(
        data: &'a mut [T],
        shape: [usize; 5],
    ) -> Option<Self> {
        let len = shape.iter().fold(1usize, |a, &d| a.saturating_mul(d));

        if len == 0 || data.len() != len {
            return None;
        }

        Some(Self { data, shape })
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        self.data
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn shape(&self) -> [usize; 5] {
        self.shape
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
