use crate::base::math::Float;

fn step_decay_impl<T: Float>(base_lr: T, step: u32, step_size: u32, gamma: T) -> Option<T> {
    if step_size == 0
        || !base_lr.is_finite()
        || base_lr <= T::ZERO
        || !gamma.is_finite()
        || gamma <= T::ZERO
    {
        return None;
    }
    let k = (step.max(1) - 1) / step_size;
    Some(base_lr * gamma.powf(T::from_i32(k as i32)))
}

fn cosine_decay_impl<T: Float>(
    base_lr: T,
    step: u32,
    total_steps: u32,
    min_lr_ratio: T,
) -> Option<T> {
    if total_steps == 0
        || !base_lr.is_finite()
        || base_lr <= T::ZERO
        || !min_lr_ratio.is_finite()
        || min_lr_ratio < T::ZERO
        || min_lr_ratio > T::ONE
    {
        return None;
    }
    let s = step.max(1).min(total_steps);
    let progress = T::from_i32(s as i32) / T::from_i32(total_steps as i32);
    let pi = T::from_f32(core::f32::consts::PI);
    let cosine = T::HALF * (T::ONE + (pi * progress).cos());
    Some(base_lr * (min_lr_ratio + (T::ONE - min_lr_ratio) * cosine))
}

pub fn step_decay<T: Float>(base_lr: T, step: u32, step_size: u32, gamma: T) -> Option<T> {
    step_decay_impl(base_lr, step, step_size, gamma)
}

pub fn cosine_decay<T: Float>(base_lr: T, step: u32, total_steps: u32, min_lr_ratio: T) -> Option<T> {
    cosine_decay_impl(base_lr, step, total_steps, min_lr_ratio)
}
