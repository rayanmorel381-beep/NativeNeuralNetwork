#[inline]
pub(crate) fn floord(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let t = x as i64 as f64;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

#[inline]
pub(crate) fn floorf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let t = x as i32 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

#[inline]
pub(crate) fn ldexpf(x: f32, exp: i32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let mant = bits & 0x007f_ffff;
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    e += exp;
    if e <= -127 {
        return 0.0;
    }
    if e >= 128 {
        return f32::INFINITY;
    }
    let new_bits = sign | (((e + 127) as u32) << 23) | mant;
    f32::from_bits(new_bits)
}

#[inline]
pub(crate) fn ldexpd(x: f64, exp: i32) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000u64;
    let mant = bits & 0x000f_ffff_ffff_ffffu64;
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    e += exp;
    if e <= -1023 {
        return 0.0;
    }
    if e >= 1024 {
        return f64::INFINITY;
    }
    let new_bits = sign | (((e + 1023) as u64) << 52) | mant;
    f64::from_bits(new_bits)
}
