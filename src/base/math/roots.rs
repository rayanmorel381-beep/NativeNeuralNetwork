#[inline]
pub fn sqrtf(x: f32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    if x.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
        return f32::NAN;
    }
    let xhalf = 0.5_f32 * x;
    let mut i = x.to_bits();
    i = 0x5f3759dfu32.wrapping_sub(i >> 1);
    let mut y = f32::from_bits(i);
    y = y * (1.5 - xhalf * y * y);
    y = y * (1.5 - xhalf * y * y);
    x * y
}

#[inline]
pub fn sqrtd(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }
    let xhalf = 0.5_f64 * x;
    let mut i = x.to_bits();
    i = 0x5fe6_ec85_e7de_30da_u64.wrapping_sub(i >> 1);
    let mut y = f64::from_bits(i);
    y = y * (1.5 - xhalf * y * y);
    y = y * (1.5 - xhalf * y * y);
    x * y
}
