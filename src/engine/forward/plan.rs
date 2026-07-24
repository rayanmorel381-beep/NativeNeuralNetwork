use crate::graph::net::layers::{LayerError, LayerPlan};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardError {
    InvalidPlan,
    ShapeMismatch,
    ScratchTooSmall,
}

pub fn validate_forward_plan(plan: &LayerPlan<'_>) -> Result<(), ForwardError> {
    match plan.validate() {
        Ok(()) => Ok(()),
        Err(LayerError::EmptyPlan | LayerError::InvalidShape | LayerError::IncompatibleChain) => {
            Err(ForwardError::InvalidPlan)
        }
        Err(LayerError::BufferTooSmall | LayerError::CountMismatch) => {
            Err(ForwardError::ScratchTooSmall)
        }
        Err(LayerError::InvalidRange) => Err(ForwardError::ShapeMismatch),
    }
}

pub fn required_single_infer_scratch(plan: &LayerPlan<'_>) -> Option<usize> {
    plan.max_width()?.checked_mul(2)
}
