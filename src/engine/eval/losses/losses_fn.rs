use crate::base::math::Float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LossKind {
    Mse,
    Mae,
    Huber { delta: f32 },
}

impl LossKind {
    pub(crate) fn from_tag(tag: u8, huber_delta: f32) -> LossKind {
        match tag {
            1 => LossKind::Mae,
            2 => LossKind::Huber { delta: huber_delta },
            _ => LossKind::Mse,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossError {
    Empty,
    ShapeMismatch,
    NonFinite,
}

fn loss_and_gradient_impl<T: Float>(
    kind_tag: u8,
    huber_delta: T,
    prediction: &[T],
    target: &[T],
    grad_out: &mut [T],
) -> Result<T, LossError> {
    if prediction.is_empty() {
        return Err(LossError::Empty);
    }
    if prediction.len() != target.len() || grad_out.len() < prediction.len() {
        return Err(LossError::ShapeMismatch);
    }

    let n = T::from_usize(prediction.len());
    let mut loss = T::ZERO;

    match kind_tag {
        0 => {
            let scale = T::TWO / n;
            for i in 0..prediction.len() {
                let diff = prediction[i] - target[i];
                loss += diff * diff;
                grad_out[i] = scale * diff;
            }
            loss /= n;
        }
        1 => {
            let inv_n = n.recip();
            for i in 0..prediction.len() {
                let diff = prediction[i] - target[i];
                loss += diff.abs();
                grad_out[i] = if diff > T::ZERO {
                    inv_n
                } else if diff < T::ZERO {
                    T::ZERO - inv_n
                } else {
                    T::ZERO
                };
            }
            loss *= inv_n;
        }
        _ => {
            let delta = huber_delta;
            if !delta.is_finite() || delta <= T::ZERO {
                return Err(LossError::NonFinite);
            }
            let inv_n = n.recip();
            for i in 0..prediction.len() {
                let diff = prediction[i] - target[i];
                let ad = diff.abs();
                if ad <= delta {
                    loss += T::HALF * diff * diff;
                    grad_out[i] = diff * inv_n;
                } else {
                    loss += delta * (ad - T::HALF * delta);
                    grad_out[i] = if diff > T::ZERO {
                        delta * inv_n
                    } else {
                        T::ZERO - delta * inv_n
                    };
                }
            }
            loss *= inv_n;
        }
    }

    if !loss.is_finite() {
        return Err(LossError::NonFinite);
    }

    Ok(loss)
}

pub fn loss_and_gradient_f32(
    kind: LossKind,
    prediction: &[f32],
    target: &[f32],
    grad_out: &mut [f32],
) -> Result<f32, LossError> {
    let (tag, delta) = match kind {
        LossKind::Mse => (0u8, 0.0f32),
        LossKind::Mae => (1u8, 0.0f32),
        LossKind::Huber { delta } => (2u8, delta),
    };
    loss_and_gradient_impl(tag, delta, prediction, target, grad_out)
}

pub fn loss_and_gradient_f64(
    kind: LossKind,
    prediction: &[f64],
    target: &[f64],
    grad_out: &mut [f64],
) -> Result<f64, LossError> {
    let (tag, delta) = match kind {
        LossKind::Mse => (0u8, 0.0f64),
        LossKind::Mae => (1u8, 0.0f64),
        LossKind::Huber { delta } => (2u8, delta as f64),
    };
    loss_and_gradient_impl(tag, delta, prediction, target, grad_out)
}
