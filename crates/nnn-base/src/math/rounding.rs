use super::elementwise;
use super::float::{Float, MathError};
use crate::scratch::typed::Buf;

pub(crate) fn round_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::round)
}

#[inline]
pub(crate) fn roundf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    // At or above 2^23 every f32 is already an integer, so the cast below would
    // only saturate and hand back a different value. Returning `x` keeps large
    // magnitudes intact.
    if x >= 8_388_608.0 || x <= -8_388_608.0 {
        return x;
    }
    // Anything strictly inside the first half step rounds to zero. `0.0 * x`
    // carries the sign of `x`, so a small negative gives -0.0 rather than the
    // +0.0 the cast would produce.
    if x < 0.5 && x > -0.5 {
        return 0.0 * x;
    }
    if x > 0.0 {
        (x + 0.5) as i32 as f32
    } else {
        (x - 0.5) as i32 as f32
    }
}

#[inline]
pub(crate) fn roundd(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    // 2^52 is where f64 stops having a fractional bit to round.
    if x >= 4_503_599_627_370_496.0 || x <= -4_503_599_627_370_496.0 {
        return x;
    }
    if x < 0.5 && x > -0.5 {
        return 0.0 * x;
    }
    if x > 0.0 {
        (x + 0.5) as i64 as f64
    } else {
        (x - 0.5) as i64 as f64
    }
}
