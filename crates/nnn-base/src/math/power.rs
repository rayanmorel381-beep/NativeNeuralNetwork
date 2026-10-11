use super::exp_log::{expd, expf, lnd, lnf};
use super::float::{Float, MathError};
use crate::scratch::typed::Buf;

pub(crate) fn pow_carve<T: Float>(
	buffer: &mut Buf<'_, T>,
	exponent: T,
) -> Result<usize, MathError> {
	Ok(T::carve_map(buffer, |value| value.powf(exponent))?)
}

/// Whether `y` is an odd integer.
///
/// Every `f32` at or above 2^24 is an even integer — the spacing there is at
/// least two, so no odd value is representable — which is why the guard returns
/// `false` instead of casting. Below it, the cast is exact and the low bit is
/// the parity.
#[inline]
fn odd_integerf(y: f32) -> bool {
    if y.abs() >= 16_777_216.0 {
        return false;
    }
    (y as i64) & 1 == 1
}

/// Whether `y` is an odd integer, for `f64`.
#[inline]
fn odd_integerd(y: f64) -> bool {
    if y.abs() >= 9_007_199_254_740_992.0 {
        return false;
    }
    (y as i64) & 1 == 1
}

/// Whether `y` is an integer at all (`fract` lives in `std`, not `core`).
#[inline]
fn integerf(y: f32) -> bool {
    y.abs() >= 8_388_608.0 || y == (y as i64) as f32
}

/// Whether `y` is an integer at all, for `f64`.
#[inline]
fn integerd(y: f64) -> bool {
    y.abs() >= 4_503_599_627_370_496.0 || y == (y as i64) as f64
}

#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }

    if y == 0.0 {
        return 1.0;
    }

    if x == 1.0 {
        return 1.0;
    }

    if x == 0.0 {
        // A negative base is only carried through by an odd integer exponent,
        // and a signed zero is a negative base.
        let negative = x.is_sign_negative() && integerf(y) && odd_integerf(y);

        return if y > 0.0 {
            if negative {
                -0.0
            } else {
                0.0
            }
        } else if negative {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }

    if x < 0.0 {
        // A real power of a negative base exists only for integer exponents;
        // an odd one keeps the sign, an even one drops it.
        if !integerf(y) {
            return f32::NAN;
        }

        let magnitude = expf(y * lnf(-x));

        return if odd_integerf(y) { -magnitude } else { magnitude };
    }

    expf(y * lnf(x))
}

#[inline]
pub(crate) fn powd(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    if y == 0.0 {
        return 1.0;
    }

    if x == 1.0 {
        return 1.0;
    }

    if x == 0.0 {
        let negative = x.is_sign_negative() && integerd(y) && odd_integerd(y);

        return if y > 0.0 {
            if negative {
                -0.0
            } else {
                0.0
            }
        } else if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }

    if x < 0.0 {
        if !integerd(y) {
            return f64::NAN;
        }

        let magnitude = expd(y * lnd(-x));

        return if odd_integerd(y) { -magnitude } else { magnitude };
    }

    expd(y * lnd(x))
}
