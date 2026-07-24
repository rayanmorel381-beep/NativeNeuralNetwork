use super::decay::{cosine_decay, step_decay};
use super::policy::{LrSchedule, ScheduleError};
use super::warmup::inv_sqrt_warmup;
use super::warmup::linear_warmup;
use crate::base::math::Float;

pub fn compute_learning_rate<T: Float>(
    base_lr: T,
    step: u32,
    schedule: LrSchedule<T>,
) -> Result<T, ScheduleError> {
    let lr = match schedule {
        LrSchedule::Constant => Some(base_lr),
        LrSchedule::StepDecay { step_size, gamma } => step_decay(base_lr, step, step_size, gamma),
        LrSchedule::Cosine {
            total_steps,
            min_lr_ratio,
        } => cosine_decay(base_lr, step, total_steps, min_lr_ratio),
        LrSchedule::LinearWarmup { warmup_steps } => linear_warmup(base_lr, step, warmup_steps),
        LrSchedule::InvSqrt { warmup_steps } => inv_sqrt_warmup(base_lr, step, warmup_steps),
    };

    let lr = lr.ok_or(ScheduleError::InvalidConfig)?;
    if !lr.is_finite() || lr <= T::ZERO {
        return Err(ScheduleError::InvalidConfig);
    }
    Ok(lr)
}
