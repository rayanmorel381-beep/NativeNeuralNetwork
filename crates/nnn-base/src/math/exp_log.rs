use super::elementwise;
use super::float::{Float, MathError};
use super::helpers::{floord, floorf, ldexpd, ldexpf};
use crate::scratch::typed::Buf;
use core::f32::consts::{LN_2, LOG2_E};
use core::f64::consts::{LN_2 as LN_2_D, LOG2_E as LOG2_E_D};

pub(crate) fn exp_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::exp)
}

pub(crate) fn ln_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::ln)
}

#[inline]
pub(crate) fn expf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let x = x.clamp(-88.0, 88.0);
    let inv_ln2: f32 = LOG2_E;
    let n = floorf(x * inv_ln2) as i32;
    let r = x - (n as f32) * LN_2;
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    let r5 = r4 * r;
    let approx = 1.0 + r + 0.5 * r2 + (1.0 / 6.0) * r3 + (1.0 / 24.0) * r4 + (1.0 / 120.0) * r5;
    ldexpf(approx, n)
}

#[inline]
pub(crate) fn expd(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    // NaN is already handled above, so this is a plain clamp to the range where
    // `2^n * poly(r)` stays finite in `f64`.
    let x = x.clamp(-709.0, 709.0);
    let inv_ln2 = LOG2_E_D;
    let n = floord(x * inv_ln2) as i32;
    let r = x - (n as f64) * LN_2_D;
    let r2 = r * r;
    let r3 = r2 * r;
    let r4 = r3 * r;
    let r5 = r4 * r;
    let approx = 1.0 + r + 0.5 * r2 + (1.0 / 6.0) * r3 + (1.0 / 24.0) * r4 + (1.0 / 120.0) * r5;
    ldexpd(approx, n)
}

#[inline]
pub(crate) fn lnf(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 23) & 0xff) as i32;
    let fraction = bits & 0x007f_ffff;
    // A subnormal has no implicit leading one, so the exponent has to come from
    // the highest set bit of the fraction instead of the (empty) exponent field.
    // Forcing the implicit one on the fraction, as before, read the whole
    // subnormal range as if it sat just above 1.0.
    let (mant_bits, e) = if exp_field != 0 {
        (fraction | 0x3f80_0000, exp_field - 127)
    } else {
        let lead = 31 - fraction.leading_zeros() as i32;
        (fraction << (23 - lead), lead - 149)
    };
    let m = f32::from_bits(mant_bits);
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let ln_m = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0);
    ln_m + (e as f32) * LN_2
}

#[inline]
pub(crate) fn lnd(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffffu64;
    let (mant_bits, e) = if exp_field != 0 {
        (fraction | 0x3ff0_0000_0000_0000u64, exp_field - 1023)
    } else {
        let lead = 63 - fraction.leading_zeros() as i32;
        (fraction << (52 - lead), lead - 1074)
    };
    let m = f64::from_bits(mant_bits);
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let ln_m = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0);
    ln_m + (e as f64) * LN_2_D
}
