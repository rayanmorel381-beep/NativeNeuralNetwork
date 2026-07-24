use core::fmt;

pub struct TensorView<'a> {
    pub data: &'a mut [f32],
    pub shape: [usize; 5],
}

impl<'a> TensorView<'a> {
    pub fn is_valid_layout(&self) -> bool {
        self.len() == self.data.len()
    }

    pub fn len(&self) -> usize {
        let mut n = 1usize;
        for &d in &self.shape {
            n = n.saturating_mul(d);
        }
        n
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn idx_linear(&self, n: usize, c: usize, d: usize, h: usize, w: usize) -> Option<usize> {
        let [n_dim, c_dim, d_dim, h_dim, w_dim] = self.shape;
        if n >= n_dim || c >= c_dim || d >= d_dim || h >= h_dim || w >= w_dim {
            return None;
        }
        let idx = n
            .checked_mul(c_dim)?
            .checked_add(c)?
            .checked_mul(d_dim)?
            .checked_add(d)?
            .checked_mul(h_dim)?
            .checked_add(h)?
            .checked_mul(w_dim)?
            .checked_add(w)?;
        if idx < self.data.len() {
            Some(idx)
        } else {
            None
        }
    }

    pub fn get(&self, n: usize, c: usize, d: usize, h: usize, w: usize) -> Option<f32> {
        let i = self.idx_linear(n, c, d, h, w)?;
        Some(self.data[i])
    }

    pub fn get_mut(
        &mut self,
        n: usize,
        c: usize,
        d: usize,
        h: usize,
        w: usize,
    ) -> Option<&mut f32> {
        let i = self.idx_linear(n, c, d, h, w)?;
        Some(&mut self.data[i])
    }

    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }
}

impl<'a> fmt::Debug for TensorView<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape)
            .finish()
    }
}
