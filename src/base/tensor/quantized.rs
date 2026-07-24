use crate::base::math::Float;

pub struct QuantizedTensor<'a, T> {
    pub q: &'a [i8],
    pub scales: &'a [T],
    pub group_size: usize,
    pub shape: [usize; 5],
}

impl<'a, T: Float> QuantizedTensor<'a, T> {
    pub fn from_qweight(q: &'a [i8], scales: &'a [T], group_size: usize, shape: [usize; 5]) -> Self {
        Self { q, scales, group_size, shape }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
    }

    pub fn dequant_into(&self, out: &mut [T]) {
        let gs = self.group_size.max(1);
        for (i, o) in out.iter_mut().enumerate() {
            if i >= self.q.len() { break; }
            let scale_idx = i / gs;
            let scale = if scale_idx < self.scales.len() { self.scales[scale_idx] } else { T::ONE };
            *o = T::from_f32(self.q[i] as f32) * scale;
        }
    }
}
