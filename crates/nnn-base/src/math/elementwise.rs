use crate::scratch::carve::CarveError;
use crate::scratch::typed::Buf;

use super::float::{Float, MathError};

pub(crate) fn unary<T: Float>(
	buffer: &mut Buf<'_, T>,
	op: fn(T) -> T,
) -> Result<usize, MathError> {
	Ok(T::carve_map(buffer, op)?)
}

pub(crate) fn unary_into<T: Float>(
	source: &Buf<'_, T>,
	target: &mut Buf<'_, T>,
	op: fn(T) -> T,
) -> Result<usize, MathError> {
	let count = T::carve_count(source)?;

	if target.len() < count {
		return Err(CarveError::SizeMismatch.into());
	}

	for (index, value) in source.as_slice()[..count].iter().enumerate() {
		target.as_mut_slice()[index] = op(*value);
	}

	Ok(count)
}

pub(crate) fn binary<T: Float>(
	left: &mut Buf<'_, T>,
	right: &Buf<'_, T>,
	op: fn(T, T) -> T,
) -> Result<usize, MathError> {
	if left.len() != right.len() {
		return Err(CarveError::SizeMismatch.into());
	}

	for (left_value, right_value) in left
		.as_mut_slice()
		.iter_mut()
		.zip(right.as_slice())
	{
		*left_value = op(*left_value, *right_value);
	}

	Ok(left.len())
}

pub(crate) fn scaled<T: Float>(
	buffer: &mut Buf<'_, T>,
	factor: T,
) -> Result<usize, MathError> {
	Ok(T::carve_map(buffer, |value| value * factor)?)
}

pub(crate) fn clamped<T: Float>(
	buffer: &mut Buf<'_, T>,
	low: T,
	high: T,
) -> Result<usize, MathError> {
	if low > high {
		return Err(CarveError::SizeMismatch.into());
	}

	Ok(T::carve_map(buffer, |value| {
		if value < low {
			low
		} else if value > high {
			high
		} else {
			value
		}
	})?)
}
