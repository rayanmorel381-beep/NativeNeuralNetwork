//! The activation layer, the third tier of the tower.
//!
//! [`ActivationKind`] names the operation; [`LmlpActivations`] is the layer that
//! runs it over a carve. The two are kept together because they are one idea —
//! an activation is a name and an in-place pass — and because the layer exists
//! only to give `base.rs` something to hold.

use super::functions::{identity, softmax_output};
use super::numeric_guard::validate_activation_outputs;
use crate::math::elementwise;
use crate::math::float::{Float, MathError};
use crate::scratch::typed::Buf;
use core::marker::PhantomData;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind {
	Identity,
	Relu,
	Sigmoid,
	Tanh,
	SoftmaxOutput,
}

impl ActivationKind {
	/// The scalar op, for the kinds that have one.
	///
	/// `SoftmaxOutput` is not a scalar function — it normalises across a whole
	/// row — so at a single point it is the identity, and the row pass goes
	/// through [`ActivationKind::apply_rows`] instead.
	pub(crate) fn op<T: Float>(self) -> fn(T) -> T {
		match self {
			ActivationKind::Identity => identity,
			ActivationKind::Relu => <T as Float>::relu,
			ActivationKind::Sigmoid => <T as Float>::sigmoid,
			ActivationKind::Tanh => <T as Float>::tanh,
			ActivationKind::SoftmaxOutput => softmax_output,
		}
	}

	pub(crate) fn apply<T: Float>(self, x: T) -> T {
		(self.op())(x)
	}

	/// Applies the activation over a buffer, treating it as a single row.
	pub(crate) fn apply_in_place<T: Float>(
		self,
		buffer: &mut Buf<'_, T>,
	) -> Result<usize, MathError> {
		self.apply_rows(buffer, 1)
	}

	/// Applies the activation over `rows` equal rows of `buffer`.
	///
	/// The row-based pass is what `SoftmaxOutput` needs; every other kind is
	/// per element and ignores the split. Keeping the entry point row-aware is
	/// what closes the gap where `SoftmaxOutput` was an identity over a buffer
	/// and a real softmax only existed in `math::float`.
	pub(crate) fn apply_rows<T: Float>(
		self,
		buffer: &mut Buf<'_, T>,
		rows: usize,
	) -> Result<usize, MathError> {
		match self {
			ActivationKind::SoftmaxOutput => {
				crate::math::float::carve_softmax_rows(buffer, rows)
			}
			_ => elementwise::unary(buffer, self.op()),
		}
	}
}

/// The activation layer, held by [`LmlpBase`].
///
/// [`LmlpBase`]: crate::base::LmlpBase
pub(crate) struct LmlpActivations<T> {
	_marker: PhantomData<T>,
}

impl<T: Float> LmlpActivations<T> {
	pub(crate) const fn new() -> Self {
		Self {
			_marker: PhantomData,
		}
	}

	pub(crate) fn apply(
		&self,
		kind: ActivationKind,
		buffer: &mut Buf<'_, T>,
	) -> Result<usize, MathError> {
		kind.apply_in_place(buffer)
	}

	pub(crate) fn apply_rows(
		&self,
		kind: ActivationKind,
		buffer: &mut Buf<'_, T>,
		rows: usize,
	) -> Result<usize, MathError> {
		kind.apply_rows(buffer, rows)
	}

	pub(crate) fn validate(
		&self,
		kind: ActivationKind,
		buffer: &Buf<'_, T>,
	) -> Result<usize, MathError> {
		validate_activation_outputs(kind, buffer)
	}
}
