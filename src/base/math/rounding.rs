#[inline]
pub fn roundf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x >= 0.0 {
        (x + 0.5) as i32 as f32
    } else {
        (x - 0.5) as i32 as f32
    }
}

#[inline]
pub fn roundd(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x >= 0.0 {
        (x + 0.5) as i64 as f64
    } else {
        (x - 0.5) as i64 as f64
    }
}
