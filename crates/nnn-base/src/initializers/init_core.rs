use crate::math::float::{F32, F64, Float};
use crate::math::helpers::{from_f32, from_f64};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitKind<T: Float> {
	Zeros,
	XavierUniform,
	HeUniform,
	Constant(T),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
	InvalidShape,
	ShapeMismatch,
	NonFinite,
}

pub(crate) fn expected_parameter_counts(layers: &[usize]) -> Option<(usize, usize)> {
	if layers.len() < 2 {
		return None;
	}

	let mut weights = 0usize;
	let mut biases = 0usize;
	for i in 0..layers.len() - 1 {
		let in_size = layers[i];
		let out_size = layers[i + 1];
		if in_size == 0 || out_size == 0 {
			return None;
		}
		weights = weights.checked_add(in_size.checked_mul(out_size)?)?;
		biases = biases.checked_add(out_size)?;
	}
	Some((weights, biases))
}

fn initialize_parameters_impl<T: Float>(
	layers: &[usize],
	weights: &mut [T],
	biases: &mut [T],
	kind: InitKind<T>,
	rng: &mut SplitMix64,
) -> Result<(), InitError> {
	let (expected_w, expected_b) =
		expected_parameter_counts(layers).ok_or(InitError::InvalidShape)?;
	if weights.len() != expected_w || biases.len() != expected_b {
		return Err(InitError::ShapeMismatch);
	}

	let mut w_off = 0usize;
	let mut b_off = 0usize;
	let six = from_f32::<T>(F32::from_native(6.0));

	for i in 0..layers.len() - 1 {
		let in_size = layers[i];
		let out_size = layers[i + 1];
		let w_len = in_size * out_size;
		let (w_slice, b_slice) = (
			&mut weights[w_off..w_off + w_len],
			&mut biases[b_off..b_off + out_size],
		);

		match kind {
			InitKind::Zeros => {
				for w in w_slice {
					*w = T::ZERO;
				}
				for b in b_slice {
					*b = T::ZERO;
				}
			}
			InitKind::Constant(value) => {
				if !value.is_finite() {
					return Err(InitError::NonFinite);
				}
				for w in w_slice {
					*w = value;
				}
				for b in b_slice {
					*b = T::ZERO;
				}
			}
			InitKind::XavierUniform => {
				let denom = from_f32::<T>(F32::from_native((in_size + out_size) as f32));
				if denom <= T::ZERO || !denom.is_finite() {
					return Err(InitError::NonFinite);
				}
				let limit = T::sqrt(six / denom);
				if !limit.is_finite() {
					return Err(InitError::NonFinite);
				}
				for w in w_slice {
					*w = random_uniform_symmetric(rng, limit);
				}
				for b in b_slice {
					*b = T::ZERO;
				}
			}
			InitKind::HeUniform => {
				let denom = from_f32::<T>(F32::from_native(in_size as f32));
				if denom <= T::ZERO || !denom.is_finite() {
					return Err(InitError::NonFinite);
				}
				let limit = T::sqrt(six / denom);
				if !limit.is_finite() {
					return Err(InitError::NonFinite);
				}
				for w in w_slice {
					*w = random_uniform_symmetric(rng, limit);
				}
				for b in b_slice {
					*b = T::ZERO;
				}
			}
		}

		w_off += w_len;
		b_off += out_size;
	}

	Ok(())
}

pub(crate) fn initialize_parameters<T: Float>(
	layers: &[usize],
	weights: &mut [T],
	biases: &mut [T],
	kind: InitKind<T>,
	seed: u64,
) -> Result<(), InitError> {
	let mut rng = SplitMix64::new(seed);
	initialize_parameters_impl(layers, weights, biases, kind, &mut rng)
}

pub(crate) fn initialize_parameters_zeros<T: Float>(
	layers: &[usize],
	weights: &mut [T],
	biases: &mut [T],
) -> Result<(), InitError> {
	let mut rng = SplitMix64::new(0);
	initialize_parameters_impl(layers, weights, biases, InitKind::Zeros, &mut rng)
}

pub(crate) fn initialize_parameters_he_uniform<T: Float>(
	layers: &[usize],
	weights: &mut [T],
	biases: &mut [T],
	seed: u64,
) -> Result<(), InitError> {
	let mut rng = SplitMix64::new(seed);
	initialize_parameters_impl(layers, weights, biases, InitKind::HeUniform, &mut rng)
}

pub(crate) fn initialize_parameters_constant<T: Float>(
	layers: &[usize],
	weights: &mut [T],
	biases: &mut [T],
	value: T,
) -> Result<(), InitError> {
	let mut rng = SplitMix64::new(0);
	initialize_parameters_impl(layers, weights, biases, InitKind::Constant(value), &mut rng)
}

fn random_uniform_symmetric<T: Float>(rng: &mut SplitMix64, limit: T) -> T {
	let u = rng.next_f64_01();
	let val = from_f64::<T>(F64::from_native(u * 2.0 - 1.0));
	val * limit
}

struct SplitMix64 {
	state: u64,
}

impl SplitMix64 {
	fn new(seed: u64) -> Self {
		Self { state: seed }
	}

	fn next_u64(&mut self) -> u64 {
		self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
		let mut z = self.state;
		z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
		z ^ (z >> 31)
	}

	fn next_f64_01(&mut self) -> f64 {
		let raw = self.next_u64() >> 11;
		raw as f64 / 9007199254740992.0f64
	}
}
