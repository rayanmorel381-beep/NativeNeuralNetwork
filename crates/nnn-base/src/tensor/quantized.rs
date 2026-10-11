use crate::math::float::{F32, Float};
use crate::math::helpers::from_f32;

pub(crate) struct QuantizedTensor<'a, T> {
    q: &'a [i8],
    scales: &'a [T],
    group_size: usize,
    shape: [usize; 5],
}

impl<'a, T: Float> QuantizedTensor<'a, T> {
    pub(crate) fn from_qweight(q: &'a [i8], scales: &'a [T], group_size: usize, shape: [usize; 5]) -> Option<Self> {
        if group_size == 0 {
            return None;
        }
        let numel = shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        })?;
        let groups = numel.checked_add(group_size - 1)?.checked_div(group_size)?;
        if q.len() != numel || scales.len() < groups {
            return None;
        }
        Some(Self { q, scales, group_size, shape })
    }

    pub(crate) fn numel(&self) -> usize {
        self.shape.iter().try_fold(1usize, |count, &dimension| {
            count.checked_mul(dimension)
        }).unwrap_or(0)
    }

    pub(crate) fn group_size(&self) -> usize { self.group_size }
    pub(crate) fn shape(&self) -> [usize; 5] { self.shape }

    pub(crate) fn dequant_into(&self, out: &mut [T]) {
        let gs = self.group_size;
        for (i, o) in out.iter_mut().enumerate() {
            if i >= self.numel() || i >= self.q.len() {
                *o = T::ZERO;
                continue;
            }
            let scale_idx = i / gs;
            let scale = if scale_idx < self.scales.len() { self.scales[scale_idx] } else { T::ONE };
            *o = from_f32::<T>(F32::from_native(self.q[i] as f32)) * scale;
        }
    }
}
