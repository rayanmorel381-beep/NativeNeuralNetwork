use crate::math::float::{F32, Float};
use crate::math::helpers::from_f32;

pub(crate) struct PackedTensor<'a> {
    q: &'a [i8],
    shape: [usize; 5],
}

impl<'a> PackedTensor<'a> {
    pub(crate) fn from_raw(q: &'a [i8], shape: [usize; 5]) -> Option<Self> {
        let numel = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        })?;
        if q.len() < numel { return None; }
        Some(Self { q, shape })
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        }).unwrap_or(0)
    }

    pub(crate) fn shape(&self) -> [usize; 5] { self.shape }

    pub(crate) fn dequant_row<T: Float>(&self, row: usize, scale: T, out: &mut [T]) {
        let n = out.len();
        let base = match row.checked_mul(n) { Some(value) => value, None => return };
        for (i, o) in out.iter_mut().enumerate() {
            let idx = base + i;
            if idx < self.q.len() {
                *o = from_f32::<T>(F32::from_native(self.q[idx] as f32)) * scale;
            } else {
                *o = T::ZERO;
            }
        }
    }
}
