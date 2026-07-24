use crate::base::math::Float;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LrSchedule<T: Float> {
    Constant,
    StepDecay { step_size: u32, gamma: T },
    Cosine { total_steps: u32, min_lr_ratio: T },
    LinearWarmup { warmup_steps: u32 },
    InvSqrt { warmup_steps: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleError {
    InvalidConfig,
}
