pub struct PackedTensor<'a> {
    pub q: &'a [i8],
    pub shape: [usize; 5],
}

impl<'a> PackedTensor<'a> {
    pub fn from_raw(q: &'a [i8], shape: [usize; 5]) -> Self {
        Self { q, shape }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().fold(1usize, |a, &d| a.saturating_mul(d))
    }

    pub fn dequant_row(&self, row: usize, scale: f32, out: &mut [f32]) {
        let n = out.len();
        let base = row * n;
        for (i, o) in out.iter_mut().enumerate() {
            let idx = base + i;
            if idx < self.q.len() {
                *o = self.q[idx] as f32 * scale;
            } else {
                *o = 0.0;
            }
        }
    }
}
