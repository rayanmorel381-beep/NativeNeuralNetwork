use super::adamw::{step_adamw, AdamwConfig};
use super::kind::{OptimizerError, OptimizerKind};
use super::sgd::step_sgd;
use crate::base::math::Float;

pub fn optimizer_state_len<T: Float>(kind: OptimizerKind<T>, param_count: usize) -> Option<usize> {
    match kind {
        OptimizerKind::Sgd { .. } => Some(param_count),
        OptimizerKind::AdamW { .. } => param_count.checked_mul(2),
    }
}

pub fn apply_optimizer_step<T: Float>(
    kind: OptimizerKind<T>,
    params: &mut [T],
    grads: &[T],
    state: &mut [T],
    learning_rate: T,
    step: u32,
) -> Result<(), OptimizerError> {
    if !learning_rate.is_finite() || learning_rate <= T::ZERO {
        return Err(OptimizerError::InvalidHyperParams);
    }
    if params.len() != grads.len() {
        return Err(OptimizerError::ShapeMismatch);
    }

    let kind = match kind {
        OptimizerKind::Sgd { momentum, nesterov } => OptimizerKind::Sgd { momentum, nesterov },
        OptimizerKind::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => OptimizerKind::adamw(beta1, beta2, eps, weight_decay),
    };

    match kind {
        OptimizerKind::Sgd { momentum, nesterov } => {
            step_sgd(params, grads, state, learning_rate, momentum, nesterov)
        }
        OptimizerKind::AdamW {
            beta1,
            beta2,
            eps,
            weight_decay,
        } => {
            if state.len() != params.len().saturating_mul(2) {
                return Err(OptimizerError::ShapeMismatch);
            }
            let (m, v) = state.split_at_mut(params.len());
            let cfg = AdamwConfig {
                learning_rate,
                step,
                beta1,
                beta2,
                eps,
                weight_decay,
            };
            step_adamw(params, grads, m, v, cfg)
        }
    }
}
