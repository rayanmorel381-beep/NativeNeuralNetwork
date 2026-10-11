use super::InitKind;
use crate::math::float::Float;

pub(crate) fn layers_are_valid(layers: &[usize]) -> bool {
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

pub(crate) fn init_kind_is_finite<T: Float>(kind: &InitKind<T>) -> bool {
	match kind {
		InitKind::Constant(v) => v.is_finite(),
		_ => true,
	}
}
