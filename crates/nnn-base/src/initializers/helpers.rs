//! The initializer layer, the fourth tier of the tower.
//!
//! The generators themselves live in [`super::init_core`]; this layer is the
//! handle `base.rs` holds. Its entry points take [`Buf`]s rather than slices, so
//! the parameters always arrive carved and the layer never names a slice the
//! scratch layer did not hand out.

use super::init_core::{
	self,
	InitError,
	InitKind,
};
use super::init_guard::{init_kind_is_finite, layers_are_valid};
use crate::math::float::Float;
use crate::scratch::typed::Buf;
use core::marker::PhantomData;

/// The initializer layer, held by [`LmlpBase`].
///
/// [`LmlpBase`]: crate::base::LmlpBase
pub(crate) struct LmlpInitializers<T> {
	_marker: PhantomData<T>,
}

impl<T: Float> LmlpInitializers<T> {
	pub(crate) const fn new() -> Self {
		Self {
			_marker: PhantomData,
		}
	}

	pub(crate) fn initialize(
		&self,
		layers: &[usize],
		weights: &mut Buf<'_, T>,
		biases: &mut Buf<'_, T>,
		kind: InitKind<T>,
		seed: u64,
	) -> Result<(), InitError> {
		if !init_kind_is_finite(&kind) {
			return Err(InitError::NonFinite);
		}
		self.check_layers(layers)?;
		init_core::initialize_parameters(
			layers,
			weights.as_mut_slice(),
			biases.as_mut_slice(),
			kind,
			seed,
		)
	}

	/// The zero scheme, as its own entry point.
	pub(crate) fn initialize_zeros(
		&self,
		layers: &[usize],
		weights: &mut Buf<'_, T>,
		biases: &mut Buf<'_, T>,
	) -> Result<(), InitError> {
		self.check_layers(layers)?;
		init_core::initialize_parameters_zeros(
			layers,
			weights.as_mut_slice(),
			biases.as_mut_slice(),
		)
	}

	/// The He-uniform scheme, as its own entry point.
	pub(crate) fn initialize_he_uniform(
		&self,
		layers: &[usize],
		weights: &mut Buf<'_, T>,
		biases: &mut Buf<'_, T>,
		seed: u64,
	) -> Result<(), InitError> {
		self.check_layers(layers)?;
		init_core::initialize_parameters_he_uniform(
			layers,
			weights.as_mut_slice(),
			biases.as_mut_slice(),
			seed,
		)
	}

	/// The constant scheme, as its own entry point.
	pub(crate) fn initialize_constant(
		&self,
		layers: &[usize],
		weights: &mut Buf<'_, T>,
		biases: &mut Buf<'_, T>,
		value: T,
	) -> Result<(), InitError> {
		self.check_layers(layers)?;
		init_core::initialize_parameters_constant(
			layers,
			weights.as_mut_slice(),
			biases.as_mut_slice(),
			value,
		)
	}

	/// The shape guard: at least two layers, none of them empty.
	fn check_layers(&self, layers: &[usize]) -> Result<(), InitError> {
		if layers_are_valid(layers) {
			Ok(())
		} else {
			Err(InitError::InvalidShape)
		}
	}
}
