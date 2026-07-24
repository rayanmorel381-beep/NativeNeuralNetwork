use crate::base::math::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferenceError {
    InvalidPlan,
    ShapeMismatch,
}

fn softmax_stable_impl<T: Float>(logits: &[T], out: &mut [T]) -> Result<(), InferenceError> {
    if logits.is_empty() || out.len() != logits.len() {
        return Err(InferenceError::ShapeMismatch);
    }

    let mut max_v = logits[0];
    for value in logits.iter().skip(1) {
        if *value > max_v {
            max_v = *value;
        }
    }

    let mut sum = T::ZERO;
    for i in 0..logits.len() {
        let e = (logits[i] - max_v).exp();
        out[i] = e;
        sum += e;
    }
    if !sum.is_finite() || sum <= T::ZERO {
        return Err(InferenceError::InvalidPlan);
    }
    let inv_sum = sum.recip();
    for value in out {
        *value *= inv_sum;
    }

    Ok(())
}

pub fn softmax_stable<T: Float>(logits: &[T], out: &mut [T]) -> Result<(), InferenceError> {
    if crate::engine::try_invoke_gpu_softmax::<T>(logits, out) {
        return Ok(());
    }
    softmax_stable_impl(logits, out)
}
