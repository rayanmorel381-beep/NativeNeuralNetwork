use crate::math::float::Float;

pub(crate) struct SparseTensor<T> {
    row_indices: *const usize,
    col_indices: *const usize,
    values: *const T,
    nnz: usize,
    shape: [usize; 2],
}

unsafe impl<T: Send> Send for SparseTensor<T> {}
unsafe impl<T: Sync> Sync for SparseTensor<T> {}

impl<T: Float> SparseTensor<T> {
    pub(crate) fn from_raw(
        row_indices: *const usize,
        col_indices: *const usize,
        values: *const T,
        nnz: usize,
        rows: usize,
        cols: usize,
    ) -> Option<Self> {
        if nnz == 0 || row_indices.is_null() || col_indices.is_null() || values.is_null() || rows == 0 || cols == 0 {
            return None;
        }
        Some(Self { row_indices, col_indices, values, nnz, shape: [rows, cols] })
    }

    pub(crate) fn spmv(&self, x: &[T], y: &mut [T]) {
        for i in 0..self.nnz {
            let r = unsafe { *self.row_indices.add(i) };
            let c = unsafe { *self.col_indices.add(i) };
            let v = unsafe { *self.values.add(i) };
            if r < self.shape[0] && c < self.shape[1] && r < y.len() && c < x.len() {
                y[r] += v * x[c];
            }
        }
    }

    pub(crate) fn nnz(&self) -> usize { self.nnz }
    pub(crate) fn shape(&self) -> [usize; 2] { self.shape }

    pub(crate) fn apply_mask(&self, scores: &mut [T]) {
        for i in 0..self.nnz {
            let c = unsafe { *self.col_indices.add(i) };
            let v = unsafe { *self.values.add(i) };
            if c < self.shape[1] && c < scores.len() && v == T::ZERO {
                scores[c] = T::NEG_INF;
            }
        }
    }
}
