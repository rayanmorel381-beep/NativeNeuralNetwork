use super::LossError;
use crate::base::math::Float;

fn cross_entropy_impl<T: Float>(
    logits: &[T],
    target_id: usize,
    out_probs: &mut [T],
) -> Result<T, LossError> {
    if logits.is_empty() {
        return Err(LossError::Empty);
    }
    if out_probs.len() < logits.len() {
        return Err(LossError::ShapeMismatch);
    }
    if target_id >= logits.len() {
        return Err(LossError::ShapeMismatch);
    }

    let mut max_v = logits[0];
    for &v in logits.iter().skip(1) {
        if v > max_v {
            max_v = v;
        }
    }

    let mut sum = T::ZERO;
    for (i, &v) in logits.iter().enumerate() {
        let e = (v - max_v).exp();
        out_probs[i] = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= T::ZERO {
        return Err(LossError::NonFinite);
    }
    let inv_sum = sum.recip();
    for p in out_probs[..logits.len()].iter_mut() {
        *p *= inv_sum;
    }

    let prob_correct = out_probs[target_id];
    if !prob_correct.is_finite() || prob_correct <= T::ZERO {
        return Err(LossError::NonFinite);
    }
    Ok(T::ZERO - prob_correct.ln())
}

fn cross_entropy_grad_impl<T: Float>(
    probs: &[T],
    target_id: usize,
    grad_out: &mut [T],
) -> Result<(), LossError> {
    if probs.is_empty() {
        return Err(LossError::Empty);
    }
    if grad_out.len() < probs.len() || target_id >= probs.len() {
        return Err(LossError::ShapeMismatch);
    }

    for (i, p) in probs.iter().enumerate() {
        grad_out[i] = if i == target_id { *p - T::ONE } else { *p };
    }
    Ok(())
}

pub fn cross_entropy<T: Float>(
    logits: &[T],
    target_id: usize,
    out_probs: &mut [T],
) -> Result<T, LossError> {
    cross_entropy_impl(logits, target_id, out_probs)
}

pub fn cross_entropy_grad<T: Float>(
    probs: &[T],
    target_id: usize,
    grad_out: &mut [T],
) -> Result<(), LossError> {
    cross_entropy_grad_impl(probs, target_id, grad_out)
}
