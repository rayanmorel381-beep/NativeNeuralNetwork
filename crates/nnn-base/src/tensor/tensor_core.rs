use core::fmt;
use crate::math::float::Float;

pub(crate) struct TensorView<'a, T: Float> {
    pub(crate) data: &'a mut [T],
    pub(crate) shape: [usize; 5],
}

impl<'a, T: Float> TensorView<'a, T> {
    pub(crate) fn is_valid_layout(&self) -> bool {
        self.checked_len() == Some(self.data.len())
    }

    pub(crate) fn len(&self) -> usize {
        self.checked_len().unwrap_or(0)
    }

    fn checked_len(&self) -> Option<usize> {
        let mut n = 1usize;
        for &d in &self.shape {
            n = n.checked_mul(d)?;
        }
        Some(n)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn idx_linear(&self, n: usize, c: usize, d: usize, h: usize, w: usize) -> Option<usize> {
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

    pub(crate) fn get(&self, n: usize, c: usize, d: usize, h: usize, w: usize) -> Option<T> {
        let i = self.idx_linear(n, c, d, h, w)?;
        Some(self.data[i])
    }

    pub(crate) fn get_mut(
        &mut self,
        n: usize,
        c: usize,
        d: usize,
        h: usize,
        w: usize,
    ) -> Option<&mut T> {
        let i = self.idx_linear(n, c, d, h, w)?;
        Some(&mut self.data[i])
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<'a, T: Float> fmt::Debug for TensorView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape)
            .finish()
    }
}
