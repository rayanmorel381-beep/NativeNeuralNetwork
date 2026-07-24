use super::InferenceError;
use crate::base::math::Float;

pub fn normalize_logits_in_place<T: Float>(logits: &mut [T]) -> Result<(), InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::ShapeMismatch);
    }

    let mut max_v = logits[0];
    for value in logits.iter().skip(1) {
        if *value > max_v {
            max_v = *value;
        }
    }

    let mut sum = T::ZERO;
    for value in logits.iter_mut() {
        let e = (*value - max_v).exp();
        *value = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= T::ZERO {
        return Err(InferenceError::InvalidPlan);
    }
    let inv_sum = sum.recip();
    for value in logits.iter_mut() {
        *value *= inv_sum;
    }
    Ok(())
}

fn argmax_index_impl<T: Float>(logits: &[T]) -> Option<usize> {
    if logits.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best = logits[0];
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    Some(best_idx)
}

pub fn argmax_index<T: Float>(logits: &[T]) -> Option<usize> {
    argmax_index_impl(logits)
}
