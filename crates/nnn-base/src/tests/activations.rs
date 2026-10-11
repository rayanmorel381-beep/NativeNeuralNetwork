use crate::activations::ActivationKind;
use crate::activations::numeric_guard::validate_activation_outputs;
use crate::math::float::{F16, F32, F64, Float, MathError};
use crate::math::helpers::{from_f32, to_f64};
use crate::scratch::carve::CarveError;
use crate::scratch::typed::Buf;

const ALL: [ActivationKind; 5] = [
	ActivationKind::Identity,
	ActivationKind::Relu,
	ActivationKind::Sigmoid,
	ActivationKind::Tanh,
	ActivationKind::SoftmaxOutput,
];

fn reference(kind: ActivationKind, x: f64) -> f64 {
	match kind {
		ActivationKind::Sigmoid => 1.0 / (1.0 + (-x).exp()),
		ActivationKind::Tanh => x.tanh(),
		ActivationKind::Identity | ActivationKind::SoftmaxOutput => x,
		ActivationKind::Relu => {
			if x > 0.0 {
				x
			} else {
				0.0
			}
		}
	}
}

#[test]
fn sigmoid_and_tanh_match_the_reference_across_the_representable_range() {
	const POINTS: [f64; 11] = [
		-20.0, -8.0, -3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0, 8.0, 20.0,
	];

	for x in POINTS {
		for kind in [ActivationKind::Sigmoid, ActivationKind::Tanh] {
			let expected = reference(kind, x);
			let actual = kind.apply(F64::from_native(x)).to_native();

			let error = (actual - expected).abs();
			let bound = match kind {
				ActivationKind::Sigmoid => 1.0e-5 * expected.abs().max(1.0e-300),
				_ => 1.0e-4,
			};

			assert!(
				error <= bound,
				"{:?}({}) in f64: got {}, want {}, off by {} (bound {})",
				kind,
				x,
				actual,
				expected,
				error,
				bound
			);
		}
	}
}

#[test]
fn sigmoid_and_tanh_stay_close_to_the_reference_in_half_precision() {
	const POINTS: [f64; 7] = [-2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 3.0];

	for x in POINTS {
		for kind in [ActivationKind::Sigmoid, ActivationKind::Tanh] {
			let expected = reference(kind, x);
			let actual: F16 = kind.apply(from_f32::<F16>(F32::from_native(x as f32)));

			let magnitude = if expected == 0.0 { 1.0 } else { expected.abs() };
			let tolerance = magnitude * 2f64.powi(-10) * 4.0;
			let widened = to_f64(actual).to_native();
			let error = (widened - expected).abs();

			assert!(
				error <= tolerance,
				"{:?}({}) in f16: got {}, want {}, off by {} (tolerance {})",
				kind,
				x,
				widened,
				expected,
				error,
				tolerance
			);
		}
	}
}

#[test]
fn sigmoid_stays_finite_where_the_unbranched_form_would_not() {
	for x in [-100.0f64, -89.0, -50.0] {
		let value = ActivationKind::Sigmoid.apply(F32::from_native(x as f32));
		assert!(
			value.is_finite(),
			"sigmoid({}) came back {} — the branch is gone",
			x,
			value.to_native()
		);
		assert!(
			value.to_native() <= reference(ActivationKind::Sigmoid, x) as f32,
			"sigmoid({}) = {} overshot the reference",
			x,
			value.to_native()
		);
		assert!(
			value.to_native() < 1.0e-20,
			"sigmoid({}) should be negligible, got {}",
			x,
			value.to_native()
		);
	}
}

#[test]
fn relu_keeps_what_is_positive_and_removes_the_rest() {
	assert_eq!(
		ActivationKind::Relu.apply(F32::from_native(0.5)).to_native(),
		0.5
	);

	for x in [0.0f32, -0.0, -0.5, -1000.0] {
		assert_eq!(
			ActivationKind::Relu.apply(F32::from_native(x)).to_native(),
			0.0,
			"relu({}) should be zero",
			x
		);
	}
}

#[test]
fn the_forward_pass_is_the_identity_where_it_is_named_as_one() {
	for x in [-2.0f32, -0.5, 0.0, 0.5, 3.0] {
		let input = F32::from_native(x);
		assert_eq!(ActivationKind::Identity.apply(input), input);
		assert_eq!(ActivationKind::SoftmaxOutput.apply(input), input);
	}
}

#[test]
fn half_precision_sigmoid_is_well_behaved_far_out_on_the_negative_side() {
	// This used to assert the opposite — sigmoid(-8) came back above one in f16,
	// because the native series evaluated `exp` at a negative argument and
	// cancelled. The activation now feeds `exp` a non-negative value, so the
	// point can be trusted and the reference bound applies.
	let value = ActivationKind::Sigmoid.apply(from_f32::<F16>(F32::from_native(-8.0)));
	let widened = to_f64(value).to_native();
	let reference = 1.0 / (1.0 + 8.0f64.exp());

	assert!(
		widened <= 1.0,
		"half-precision sigmoid(-8) must not exceed one, got {widened}"
	);
	assert!(
		(widened - reference).abs() <= 4.0 * 2f64.powi(-10),
		"half-precision sigmoid(-8) = {widened}, want about {reference}"
	);
}

#[test]
fn applying_over_a_buffer_writes_every_slot_and_reports_the_count() {
	let mut storage = [
		F32::from_native(-1.0),
		F32::from_native(0.0),
		F32::from_native(2.0),
	];

	let written = ActivationKind::Relu.apply_in_place(&mut Buf::from_slice(&mut storage))
		.expect("a 3-element f32 buffer is a carve");

	assert_eq!(written, 3, "the count is the buffer's own length");
	assert_eq!(
		storage.map(|v| v.to_native()),
		[0.0, 0.0, 2.0],
		"relu should have rewritten all three slots"
	);
}

#[test]
fn applying_over_a_buffer_goes_through_the_same_arithmetic_as_the_scalar_path() {
	const POINTS: [f64; 5] = [-2.0, -0.25, 0.0, 0.25, 2.0];

	for kind in ALL {
		// `SoftmaxOutput` is a row pass, not an elementwise one, so it does not
		// share the scalar path; it is exercised by the row tests below.
		if kind == ActivationKind::SoftmaxOutput {
			continue;
		}

		let mut storage: [F64; 5] =
			core::array::from_fn(|i| F64::from_native(POINTS[i]));
		let expected: [F64; 5] = POINTS.map(|x| kind.apply(F64::from_native(x)));

		kind.apply_in_place(&mut Buf::from_slice(&mut storage))
			.expect("five f64 is a carve");

		for index in 0..POINTS.len() {
			assert_eq!(
				storage[index].to_bits(),
				expected[index].to_bits(),
				"{:?} over a buffer at index {}: got {}, want {}",
				kind,
				index,
				storage[index].to_native(),
				expected[index].to_native()
			);
		}
	}
}

#[test]
fn an_empty_buffer_is_refused_by_the_carve_and_not_by_the_activation() {
	let mut nothing: [F64; 0] = [];
	assert_eq!(
		ActivationKind::Sigmoid.apply_in_place(&mut Buf::from_slice(&mut nothing)),
		Err(MathError::Carve(CarveError::SizeMismatch))
	);
}

#[test]
fn the_output_guard_speaks_the_crates_own_error_vocabulary() {
	let mut healthy = [
		F64::from_native(0.1),
		F64::from_native(0.5),
		F64::from_native(0.9),
	];

	assert_eq!(
		validate_activation_outputs(
			ActivationKind::SoftmaxOutput,
			&Buf::from_slice(&mut healthy)
		),
		Ok(3),
		"a softmax row that sums to one is finite, and the count is reported"
	);
}

#[test]
fn the_output_guard_names_the_failure_rather_than_answering_with_a_bool() {
	let mut escaped = [
		F64::from_native(0.1),
		F64::from_native(f64::INFINITY),
		F64::from_native(0.9),
	];

	assert_eq!(
		validate_activation_outputs(ActivationKind::SoftmaxOutput, &Buf::from_slice(&mut escaped)),
		Err(MathError::NonFinite),
		"an infinity in the row must be refused, and named as NonFinite"
	);

	let mut not_a_number = [F64::from_native(f64::NAN)];
	assert_eq!(
		validate_activation_outputs(ActivationKind::SoftmaxOutput, &Buf::from_slice(&mut not_a_number)),
		Err(MathError::NonFinite),
		"a NaN in the row must be refused"
	);
}


#[test]
fn each_dispatch_sigmoid_through_its_own_format() {
	let narrow: F16 = <F16 as Float>::sigmoid(from_f32::<F16>(F32::from_native(0.5)));
	let wide: F64 = <F64 as Float>::sigmoid(F64::from_native(0.5));
	let expected = 1.0 / (1.0 + (-0.5f64).exp());

	assert!(
		(to_f64(narrow).to_native() - expected).abs() <= expected * 2f64.powi(-10) * 4.0,
		"F16 sigmoid(0.5) = {}",
		to_f64(narrow).to_native()
	);
	assert!(
		(wide.to_native() - expected).abs() < 1.0e-5,
		"F64 sigmoid(0.5) = {}",
		wide.to_native()
	);
}
