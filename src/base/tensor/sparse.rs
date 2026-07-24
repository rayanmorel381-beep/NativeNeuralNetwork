use crate::base::math::Float;

pub struct SparseTensor<T> {
    pub row_indices: *const usize,
    pub col_indices: *const usize,
    pub values: *const T,
    pub nnz: usize,
    pub shape: [usize; 2],
}

unsafe impl<T: Send> Send for SparseTensor<T> {}
unsafe impl<T: Sync> Sync for SparseTensor<T> {}

impl<T: Float> SparseTensor<T> {
    pub fn from_raw(
        row_indices: *const usize,
        col_indices: *const usize,
        values: *const T,
        nnz: usize,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self { row_indices, col_indices, values, nnz, shape: [rows, cols] }
    }

    pub fn spmv(&self, x: &[T], y: &mut [T]) {
        for i in 0..self.nnz {
            let r = unsafe { *self.row_indices.add(i) };
            let c = unsafe { *self.col_indices.add(i) };
            let v = unsafe { *self.values.add(i) };
            if r < y.len() && c < x.len() {
                y[r] += v * x[c];
            }
        }
    }

    pub fn apply_mask(&self, scores: &mut [T]) {
        for i in 0..self.nnz {
            let c = unsafe { *self.col_indices.add(i) };
            let v = unsafe { *self.values.add(i) };
            if c < scores.len() && v == T::ZERO {
                scores[c] = T::NEG_INF;
            }
        }
    }
}
