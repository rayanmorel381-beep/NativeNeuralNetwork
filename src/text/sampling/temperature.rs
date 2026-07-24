use super::SamplingError;
use crate::base::math::Float;

pub fn softmax_temperature<T: Float>(
    logits: &mut [T],
    temperature: T,
) -> Result<(), SamplingError> {
    if logits.is_empty() {
        return Err(SamplingError::Empty);
    }
    if !temperature.is_finite() || temperature <= T::ZERO {
        return Err(SamplingError::InvalidParameter);
    }

    let inv_temp = temperature.recip();
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }
    crate::engine::infer::inference::normalize_logits_in_place(logits).map_err(|_| SamplingError::InvalidParameter)
}
