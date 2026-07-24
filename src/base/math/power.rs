use crate::base::math::exp_log::{expd, expf, lnd, lnf};

#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    if x.is_nan() || y.is_nan() {
        return f32::NAN;
    }

    if x == 0.0 {
        return if y > 0.0 {
            0.0
        } else if y == 0.0 {
            1.0
        } else {
            f32::INFINITY
        };
    }

    if x == 1.0 {
        return 1.0;
    }

    if y == 0.0 {
        return 1.0;
    }

    if x < 0.0 {
        return f32::NAN;
    }

    expf(y * lnf(x))
}

#[inline]
pub fn powd(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if y > 0.0 {
            0.0
        } else if y == 0.0 {
            1.0
        } else {
            f64::INFINITY
        };
    }
    if x == 1.0 {
        return 1.0;
    }
    if y == 0.0 {
        return 1.0;
    }
    if x < 0.0 {
        return f64::NAN;
    }
    expd(y * lnd(x))
}
