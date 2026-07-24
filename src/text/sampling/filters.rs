use super::SamplingError;
use crate::base::math::Float;

fn top_k_mask_impl<T: Float>(logits: &mut [T], k: usize, mask_value: T) -> Result<(), SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::Empty);
    }
    if k == 0 || k > logits.len() || !mask_value.is_finite() {
        return Err(SamplingError::InvalidParameter);
    }

    let mut threshold = T::NEG_INF;
    for _ in 0..k {
        let mut best = T::NEG_INF;
        for &v in logits.iter() {
            if (threshold == T::NEG_INF || v < threshold) && v > best {
                best = v;
            }
        }
        threshold = best;
    }

    for v in logits.iter_mut() {
        if *v < threshold {
            *v = mask_value;
        }
    }
    Ok(())
}

fn top_p_cutoff_impl<T: Float>(probabilities: &[T], p: T) -> Result<usize, SamplingError> {
    if probabilities.is_empty() {
        return Err(SamplingError::Empty);
    }
    if !p.is_finite() || p <= T::ZERO || p > T::ONE {
        return Err(SamplingError::InvalidParameter);
    }

    let mut cumulative = T::ZERO;
    for (i, &v) in probabilities.iter().enumerate() {
        if !v.is_finite() || v < T::ZERO {
            return Err(SamplingError::InvalidParameter);
        }
        cumulative += v;
        if cumulative >= p {
            return Ok(i + 1);
        }
    }
    Ok(probabilities.len())
}

pub fn top_k_mask<T: Float>(logits: &mut [T], k: usize, mask_value: T) -> Result<(), SamplingError> {
    top_k_mask_impl(logits, k, mask_value)
}

pub fn top_p_cutoff<T: Float>(probabilities: &[T], p: T) -> Result<usize, SamplingError> {
    top_p_cutoff_impl(probabilities, p)
}
