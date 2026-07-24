use crate::base::math::exp_log::{expd, expf};
use crate::base::math::rounding::{roundd, roundf};
use core::f32::consts::PI;
use core::f64::consts::PI as PI_D;

#[inline]
pub fn sinf(mut x: f32) -> f32 {
    let two_pi = 2.0 * PI;
    x = x - roundf(x / two_pi) * two_pi;
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0
}

#[inline]
pub fn cosf(mut x: f32) -> f32 {
    let two_pi = 2.0 * PI;
    x = x - roundf(x / two_pi) * two_pi;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0
}

#[inline]
pub fn tanhf(x: f32) -> f32 {
    let e2 = expf(2.0 * x);
    (e2 - 1.0) / (e2 + 1.0)
}

#[inline]
pub fn tanhd(x: f64) -> f64 {
    let e2 = expd(2.0 * x);
    (e2 - 1.0) / (e2 + 1.0)
}

#[inline]
pub fn sind(x: f64) -> f64 {
    let mut x = x;
    let two_pi = 2.0 * PI_D;
    x = x - roundd(x / two_pi) * two_pi;
    let x2 = x * x;
    let x3 = x2 * x;
    let x5 = x3 * x2;
    let x7 = x5 * x2;
    x - x3 / 6.0 + x5 / 120.0 - x7 / 5040.0
}

#[inline]
pub fn cosd(x: f64) -> f64 {
    let mut x = x;
    let two_pi = 2.0 * PI_D;
    x = x - roundd(x / two_pi) * two_pi;
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0
}
