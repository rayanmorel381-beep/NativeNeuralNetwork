use crate::base::math::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradientError {
    Empty,
    InvalidThreshold,
}

fn l2_norm_impl<T: Float>(values: &[T]) -> Result<T, GradientError> {
    if values.is_empty() {
        return Err(GradientError::Empty);
    }
    let mut sum = T::ZERO;
    for &v in values {
        sum += v * v;
    }
    Ok(sum.sqrt())
}

fn clip_by_global_norm_impl<T: Float>(values: &mut [T], max_norm: T) -> Result<T, GradientError> {
    if values.is_empty() {
        return Err(GradientError::Empty);
    }
    if !max_norm.is_finite() || max_norm <= T::ZERO {
        return Err(GradientError::InvalidThreshold);
    }

    let norm = l2_norm_impl(values)?;
    if norm > max_norm {
        let scale = max_norm / norm;
        for v in values.iter_mut() {
            *v *= scale;
        }
    }
    Ok(norm)
}

pub fn l2_norm<T: Float>(values: &[T]) -> Result<T, GradientError> {
    l2_norm_impl(values)
}

pub fn clip_by_global_norm<T: Float>(values: &mut [T], max_norm: T) -> Result<T, GradientError> {
    clip_by_global_norm_impl(values, max_norm)
}

pub fn all_finite<T: Float>(values: &[T]) -> bool {
    values.iter().all(|v| v.is_finite())
}
