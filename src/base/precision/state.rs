use super::types::Precision;
use core::sync::atomic::{AtomicU8, Ordering};

static PRECISION: AtomicU8 = AtomicU8::new(Precision::F32 as u8);

pub fn set_precision(p: Precision) {
    PRECISION.store(p as u8, Ordering::SeqCst);
}

pub fn get_precision() -> Precision {
    match PRECISION.load(Ordering::SeqCst) {
        0 => Precision::F16,
        1 => Precision::BF16,
        3 => Precision::F64,
        _ => Precision::F32,
    }
}
