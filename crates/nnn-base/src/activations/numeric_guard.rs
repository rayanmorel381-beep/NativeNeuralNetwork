use super::ActivationKind;
use crate::math::float::{Float, MathError};
use crate::scratch::typed::Buf;

pub(crate) fn validate_activation_outputs<T: Float>(
	kind: ActivationKind,
	buffer: &Buf<'_, T>,
) -> Result<usize, MathError> {
	let count = T::carve_count(buffer)?;
	let op = kind.op();

	for index in 0..count {
		if !op(T::carve_read(buffer, index)?).is_finite() {
			return Err(MathError::NonFinite);
		}
	}

	Ok(count)
}