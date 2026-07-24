use super::{estimate_runtime_memory, RuntimeError, RuntimeEstimate, RuntimeProfile};
use crate::format::model_config::TransformerConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetFit {
    pub fits: bool,
    pub required_bytes: usize,
    pub budget_bytes: usize,
    pub headroom_bytes: usize,
}

pub fn check_runtime_budget(
    config: &TransformerConfig,
    profile: RuntimeProfile,
    budget_bytes: usize,
) -> Result<BudgetFit, RuntimeError> {
    let estimate = estimate_runtime_memory(config, profile)?;
    Ok(fit_from_estimate(estimate, budget_bytes))
}

pub fn fit_from_estimate(estimate: RuntimeEstimate, budget_bytes: usize) -> BudgetFit {
    if estimate.total_bytes <= budget_bytes {
        BudgetFit {
            fits: true,
            required_bytes: estimate.total_bytes,
            budget_bytes,
            headroom_bytes: budget_bytes - estimate.total_bytes,
        }
    } else {
        BudgetFit {
            fits: false,
            required_bytes: estimate.total_bytes,
            budget_bytes,
            headroom_bytes: 0,
        }
    }
}
