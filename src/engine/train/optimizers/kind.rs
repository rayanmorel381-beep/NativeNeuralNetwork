#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OptimizerKind<T: crate::base::math::Float> {
    Sgd {
        momentum: T,
        nesterov: bool,
    },
    AdamW {
        beta1: T,
        beta2: T,
        eps: T,
        weight_decay: T,
    },
}

impl<T: crate::base::math::Float> OptimizerKind<T> {
    pub fn adamw(beta1: T, beta2: T, eps: T, weight_decay: T) -> Self {
        Self::AdamW { beta1, beta2, eps, weight_decay }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OptimizerError {
    InvalidHyperParams,
    ShapeMismatch,
    StepOverflow,
}
