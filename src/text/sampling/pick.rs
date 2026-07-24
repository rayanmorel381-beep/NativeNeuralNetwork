use super::SamplingError;
use crate::base::math::Float;

fn sample_from_cumulative_impl<T: Float>(
    probabilities: &[T],
    threshold: T,
) -> Result<usize, SamplingError> {
    if probabilities.is_empty() {
        return Err(SamplingError::Empty);
    }
    if !threshold.is_finite() || threshold < T::ZERO || threshold > T::ONE {
        return Err(SamplingError::InvalidParameter);
    }

    let mut cumulative = T::ZERO;
    for (i, &p) in probabilities.iter().enumerate() {
        if !p.is_finite() || p < T::ZERO {
            return Err(SamplingError::InvalidParameter);
        }
        cumulative += p;
        if cumulative >= threshold {
            return Ok(i);
        }
    }
    Ok(probabilities.len() - 1)
}

pub fn argmax_sample<T: Float>(probabilities: &[T]) -> Result<usize, SamplingError> {
    crate::engine::infer::inference::argmax_index(probabilities).ok_or(SamplingError::Empty)
}

pub fn sample_from_cumulative<T: Float>(
    probabilities: &[T],
    threshold: T,
) -> Result<usize, SamplingError> {
    sample_from_cumulative_impl(probabilities, threshold)
}
