use crate::base::math::helpers::{floord, floorf, ldexpd, ldexpf};
use core::f32::consts::{LN_2, LOG2_E};
use core::f64::consts::{LN_2 as LN_2_D, LOG2_E as LOG2_E_D};

#[inline]
pub fn expf(x: f32) -> f32 {
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
pub fn expd(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let x = if x.is_nan() {
        x
    } else {
        x.clamp(-709.0, 709.0)
    };
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
pub fn lnf(x: f32) -> f32 {
    if x <= 0.0 {
        return f32::NAN;
    }
    let bits = x.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    let mant_bits = (bits & 0x007f_ffff) | 0x3f80_0000;
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
pub fn lnd(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    let mant_bits = (bits & 0x000f_ffff_ffff_ffffu64) | 0x3ff0_0000_0000_0000u64;
    let m = f64::from_bits(mant_bits);
    let y = (m - 1.0) / (m + 1.0);
    let y2 = y * y;
    let y3 = y2 * y;
    let y5 = y3 * y2;
    let y7 = y5 * y2;
    let ln_m = 2.0 * (y + y3 / 3.0 + y5 / 5.0 + y7 / 7.0);
    ln_m + (e as f64) * LN_2_D
}
