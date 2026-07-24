use crate::base::initializers::{InitKindF32, InitKindF64};

pub fn layers_are_valid(layers: &[usize]) -> bool {
    if layers.len() < 2 {
        return false;
    }
    for &dim in layers {
        if dim == 0 {
            return false;
        }
    }
    true
}

pub fn init_kind_is_finite_f32(kind: InitKindF32) -> bool {
    match kind {
        InitKindF32::Constant(v) => v.is_finite(),
        _ => true,
    }
}

pub fn init_kind_is_finite_f64(kind: InitKindF64) -> bool {
    match kind {
        InitKindF64::Constant(v) => v.is_finite(),
        _ => true,
    }
}
