use super::elementwise;
use super::exp_log::{expd, expf};
use super::float::{Float, MathError};
use super::rounding::roundd;
use crate::scratch::typed::Buf;
use core::f64::consts::{FRAC_2_PI, FRAC_PI_2};

pub(crate) fn sin_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::sin)
}

pub(crate) fn cos_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::cos)
}

pub(crate) fn tanh_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::tanh)
}

// The argument is first reduced to the nearest multiple of pi/2, which lands it
// in [-pi/4, pi/4] and leaves the quadrant to pick a sign and a kernel. That is
// the whole reason the kernels below stay short: a Taylor series on this range
// converges fast, where the old reduce-to-[-pi, pi] plus four terms was about
// 0.08 off at the far end. The reduction runs in `f64` even for `f32` inputs so
// the subtraction keeps its digits.
#[inline]
fn reduce_half_pi(x: f64) -> (f64, i32) {
    let n = roundd(x * FRAC_2_PI);
    let r = x - n * FRAC_PI_2;
    (r, (n as i64).rem_euclid(4) as i32)
}

#[inline]
fn sin_small(x: f64) -> f64 {
    let x2 = x * x;
    x * (1.0
        + x2 * (-1.0 / 6.0
            + x2 * (1.0 / 120.0
                + x2 * (-1.0 / 5040.0 + x2 * (1.0 / 362880.0)))))
}

#[inline]
fn cos_small(x: f64) -> f64 {
    let x2 = x * x;
    1.0 + x2 * (-0.5
        + x2 * (1.0 / 24.0
            + x2 * (-1.0 / 720.0 + x2 * (1.0 / 40320.0))))
}

#[inline]
pub(crate) fn sinf(x: f32) -> f32 {
    let (r, quadrant) = reduce_half_pi(x as f64);

    let value = match quadrant {
        0 => sin_small(r),
        1 => cos_small(r),
        2 => -sin_small(r),
        _ => -cos_small(r),
    };

    value as f32
}

#[inline]
pub(crate) fn cosf(x: f32) -> f32 {
    let (r, quadrant) = reduce_half_pi(x as f64);

    let value = match quadrant {
        0 => cos_small(r),
        1 => -sin_small(r),
        2 => -cos_small(r),
        _ => sin_small(r),
    };

    value as f32
}

#[inline]
pub(crate) fn tanhf(x: f32) -> f32 {
    let e2 = expf(2.0 * x);
    (e2 - 1.0) / (e2 + 1.0)
}

#[inline]
pub(crate) fn tanhd(x: f64) -> f64 {
    let e2 = expd(2.0 * x);
    (e2 - 1.0) / (e2 + 1.0)
}

#[inline]
pub(crate) fn sind(x: f64) -> f64 {
    let (r, quadrant) = reduce_half_pi(x);

    match quadrant {
        0 => sin_small(r),
        1 => cos_small(r),
        2 => -sin_small(r),
        _ => -cos_small(r),
    }
}

#[inline]
pub(crate) fn cosd(x: f64) -> f64 {
    let (r, quadrant) = reduce_half_pi(x);

    match quadrant {
        0 => cos_small(r),
        1 => -sin_small(r),
        2 => -cos_small(r),
        _ => sin_small(r),
    }
}
