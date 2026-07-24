use super::LoraError;
use crate::base::math::Float;
use crate::engine::runtime::{dot, axpy};

pub fn apply_lora_delta<T: Float>(
    base_weight: &mut [T],
    rows: usize,
    cols: usize,
    a: &[T],
    b: &[T],
    rank: usize,
    alpha: T,
) -> Result<(), LoraError> {
    if !alpha.is_finite() || rank == 0 {
        return Err(LoraError::ShapeMismatch);
    }
    if base_weight.len() != rows * cols || a.len() != rank * cols || b.len() != rows * rank {
        return Err(LoraError::ShapeMismatch);
    }
    let scale = alpha / T::from_usize(rank);
    for r in 0..rows {
        let brow = &b[r * rank..r * rank + rank];
        for k in 0..rank {
            let bk = brow[k] * scale;
            if bk.abs() > T::ZERO {
                let arow = &a[k * cols..k * cols + cols];
                axpy(&mut base_weight[r * cols..r * cols + cols], bk, arow, cols);
            }
        }
    }
    Ok(())
}

pub fn lora_forward_delta<T: Float>(
    x: &[T],
    a: &[T],
    b: &[T],
    out: &mut [T],
    tmp: &mut [T],
    rows: usize,
    in_d: usize,
    out_d: usize,
    rank: usize,
    alpha: T,
) -> Result<(), LoraError> {
    if !alpha.is_finite() || rank == 0 {
        return Err(LoraError::ShapeMismatch);
    }
    if x.len() < rows * in_d
        || a.len() < rank * in_d
        || b.len() < out_d * rank
        || out.len() < rows * out_d
        || tmp.len() < rows * rank
    {
        return Err(LoraError::ShapeMismatch);
    }
    let scale = alpha / T::from_usize(rank);
    for row in 0..rows {
        let xr = &x[row * in_d..row * in_d + in_d];
        let tr = &mut tmp[row * rank..row * rank + rank];
        for k in 0..rank {
            tr[k] = dot(xr, &a[k * in_d..k * in_d + in_d], in_d);
        }
    }
    for row in 0..rows {
        let tr = &tmp[row * rank..row * rank + rank];
        let or_slice = &mut out[row * out_d..row * out_d + out_d];
        for o in 0..out_d {
            let delta = dot(tr, &b[o * rank..o * rank + rank], rank) * scale;
            or_slice[o] += delta;
        }
    }
    Ok(())
}
