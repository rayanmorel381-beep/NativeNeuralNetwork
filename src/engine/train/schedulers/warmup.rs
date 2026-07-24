use crate::base::math::Float;

fn linear_warmup_impl<T: Float>(base_lr: T, step: u32, warmup_steps: u32) -> Option<T> {
    if warmup_steps == 0 || !base_lr.is_finite() || base_lr <= T::ZERO {
        return None;
    }
    let s = step.max(1).min(warmup_steps);
    Some(base_lr * (T::from_i32(s as i32) / T::from_i32(warmup_steps as i32)))
}

pub fn linear_warmup<T: Float>(base_lr: T, step: u32, warmup_steps: u32) -> Option<T> {
    linear_warmup_impl(base_lr, step, warmup_steps)
}

fn inv_sqrt_warmup_impl<T: Float>(base_lr: T, step: u32, warmup_steps: u32) -> Option<T> {
    if warmup_steps == 0 || !base_lr.is_finite() || base_lr <= T::ZERO {
        return None;
    }
    let one = T::from_i32(1);
    let w = T::from_i32(warmup_steps as i32);
    let s = T::from_i32(step.max(1) as i32);
    let w_sqrt = w.sqrt();
    let decay = one / s.sqrt();
    let warm = s / (w * w_sqrt);
    let scale = w_sqrt * decay.min(warm);
    let lr = base_lr * scale;
    if lr.is_finite() && lr > T::ZERO {
        Some(lr)
    } else {
        None
    }
}

pub fn inv_sqrt_warmup<T: Float>(base_lr: T, step: u32, warmup_steps: u32) -> Option<T> {
    inv_sqrt_warmup_impl(base_lr, step, warmup_steps)
}
