use super::elementwise;
use super::exp_log::{expd, expf};
use super::float::{Float, MathError};
use crate::scratch::typed::Buf;

pub(crate) fn relu_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::relu)
}

pub(crate) fn sigmoid_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::sigmoid)
}

#[inline]
pub(crate) fn sigmoidf(x: f32) -> f32 {
	if x >= 0.0 {
		1.0 / (1.0 + expf(-x))
	} else {
		let z = expf(x);
		z / (1.0 + z)
	}
}

#[inline]
pub(crate) fn sigmoidd(x: f64) -> f64 {
	if x >= 0.0 {
		1.0 / (1.0 + expd(-x))
	} else {
		let z = expd(x);
		z / (1.0 + z)
	}
}
