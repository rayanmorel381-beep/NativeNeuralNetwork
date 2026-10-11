use crate::math::float::Float;

pub(crate) fn identity<T: Float>(x: T) -> T {
	x
}

/// The scalar form of the softmax output activation.
///
/// A softmax has no per-element definition: it is the identity here so that the
/// scalar path stays total, and the real, row-wise pass lives in
/// [`ActivationKind::apply_rows`](super::ActivationKind::apply_rows).
pub(crate) fn softmax_output<T: Float>(x: T) -> T {
	x
}
