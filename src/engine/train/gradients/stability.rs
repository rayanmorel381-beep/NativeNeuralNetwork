use crate::base::math::Float;

fn has_nan_impl<T: Float>(values: &[T]) -> bool {
    values.iter().any(|v| v.is_nan())
}

fn has_inf_impl<T: Float>(values: &[T]) -> bool {
    values.iter().any(|v| !v.is_finite())
}

fn within_abs_bound_impl<T: Float>(values: &[T], bound: T) -> bool {
    if !bound.is_finite() || bound < T::ZERO {
        return false;
    }
    values.iter().all(|v| v.abs() <= bound)
}

pub fn has_nan_f32(values: &[f32]) -> bool {
    has_nan_impl(values)
}

pub fn has_nan_f64(values: &[f64]) -> bool {
    has_nan_impl(values)
}

pub fn has_inf_f32(values: &[f32]) -> bool {
    has_inf_impl(values)
}

pub fn has_inf_f64(values: &[f64]) -> bool {
    has_inf_impl(values)
}

pub fn within_abs_bound_f32(values: &[f32], bound: f32) -> bool {
    within_abs_bound_impl(values, bound)
}

pub fn within_abs_bound_f64(values: &[f64], bound: f64) -> bool {
    within_abs_bound_impl(values, bound)
}
