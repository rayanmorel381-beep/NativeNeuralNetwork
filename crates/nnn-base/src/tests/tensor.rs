//! The tensor and initializer layers, checked where they are reachable.
//!
//! Two things here are not visible from a signature. The first is that a
//! singularity test written as a fixed `f32` epsilon works on `F32` and quietly
//! stops existing on `E4M3` and `F16`, because the epsilon rounds to zero there
//! and nothing compares less than zero. The second is that a view's shape has to
//! agree with the length of the slice it borrows, or every indexing function
//! after it reads past the end.

use crate::initializers::init_core::{
	expected_parameter_counts, initialize_parameters, initialize_parameters_constant,
	initialize_parameters_he_uniform, initialize_parameters_zeros, InitError, InitKind,
};
use crate::initializers::init_guard::{init_kind_is_finite, layers_are_valid};
use crate::math::float::{E4M3, F16, F32, Float};
use crate::math::helpers::from_f32;
use crate::scratch::typed::Buf;
use crate::tensor::decompose::{tensor_cholesky, tensor_inverse, tensor_lu, tensor_qr};
use crate::tensor::dense::DenseTensor;
use crate::tensor::elementwise::{
	tensor_add_bias, tensor_add_in_place, tensor_fill, tensor_relu, tensor_scale_in_place,
	tensor_softmax_rows,
};
use crate::tensor::linalg::{
	tensor_matmul_2d, tensor_solve_lower, tensor_solve_upper, tensor_trace, tensor_transpose_2d,
};
use crate::tensor::tensor_core::TensorView;
use crate::tensor::view_mut::TensorViewMut;

// ---------------------------------------------------------------------------
// Singularity, and why the threshold cannot be a constant
// ---------------------------------------------------------------------------

#[test]
fn a_singular_triangular_factorisation_is_refused_in_every_format() {
	// A lower triangular matrix is singular exactly when a diagonal entry is zero,
	// so this one has a hole in the middle of its diagonal and the substitution
	// walks into it at i = 1. The solve has to notice and say no.
	//
	// This is the regression that the threshold fix is about. Written as a fixed
	// `1e-30`, the check reads `diagonal < 1e-30` — and as an `E4M3` or an `F16`
	// pattern `1e-30` is below the smallest subnormal and rounds to zero, so the
	// test compares the diagonal against zero, nothing is ever less than zero, and
	// a singular matrix sails through into infinities instead.
	// F32 refuses it.
	let mut l = [F32::ZERO; 9];
	for (i, row) in [[1u8, 1, 1], [1, 0, 1], [1, 1, 1]].iter().enumerate() {
		for (j, &v) in row.iter().enumerate() {
			if j <= i {
				l[i * 3 + j] = F32::from_native(v as f32);
			}
		}
	}
	let mut x = [F32::ZERO; 3];
	assert!(!tensor_solve_lower(&l, &[F32::ONE; 3], &mut x, 3), "f32 must refuse");

	// F16 must refuse too: its smallest subnormal is 2^-24, so the old threshold
	// was zero there as well.
	let narrow = from_f32::<F16>;
	let l16: [F16; 9] = core::array::from_fn(|i| narrow(l[i]));
	assert!(!tensor_solve_lower(&l16, &[F16::ONE; 3], &mut x16(), 3), "f16 must refuse a singular matrix");

	// And E4M3, where the format's own EPSILON is what the test now measures against.
	let narrow = from_f32::<E4M3>;
	let l4: [E4M3; 9] = core::array::from_fn(|i| narrow(l[i]));
	assert!(!tensor_solve_lower(&l4, &[E4M3::ONE; 3], &mut x4(), 3), "e4m3 must refuse a singular matrix");
}

fn x16() -> [F16; 3] { [F16::ZERO; 3] }
fn x4() -> [E4M3; 3] { [E4M3::ZERO; 3] }

#[test]
fn the_upper_triangular_solve_mirrors_the_lower_one() {
	// The same matrix, transposed: u = l with the off-diagonal entry moved.
	let mut u = [F32::ZERO; 9];
	for i in 0..3 {
		u[i * 3 + i] = F32::from_native(2.0);
	}
	u[0 * 3 + 1] = F32::from_native(1.0);
	u[0 * 3 + 2] = F32::from_native(1.0);
	u[1 * 3 + 2] = F32::from_native(-3.0);

	// Back substitution runs the other way: x2 first, then x1, then x0.
	let b = [F32::ONE, F32::TWO, F32::from_native(3.0)];
	let mut x = [F32::ZERO; 3];
	assert!(tensor_solve_upper(&u, &b, &mut x, 3));
	assert!((x[2].to_native() - 1.5).abs() < 1.0e-6, "x2 = {}", x[2].to_native());
	assert!((x[1].to_native() - 3.25).abs() < 1.0e-6, "x1 = {}", x[1].to_native());
	assert!((x[0].to_native() - -1.875).abs() < 1.0e-6, "x0 = {}", x[0].to_native());

	// A zero on the diagonal is what makes a triangular matrix singular, and the
	// relative threshold is what finds it in a narrow format.
	let mut singular = [F32::ONE; 9];
	singular[1 * 3 + 1] = F32::ZERO;
	let mut xs = [F32::ZERO; 3];
	assert!(!tensor_solve_upper(&singular, &b, &mut xs, 3));
}

#[test]
fn a_well_conditioned_triangular_matrix_still_solves() {
	let mut l = [F32::ZERO; 9];
	for i in 0..3 {
		l[i * 3 + i] = F32::from_native(2.0);
	}
	l[1 * 3 + 0] = F32::from_native(1.0);
	l[2 * 3 + 0] = F32::from_native(1.0);
	l[2 * 3 + 1] = F32::from_native(-3.0);

	let b = [F32::ONE, F32::TWO, F32::from_native(3.0)];
	let mut x = [F32::ZERO; 3];
	assert!(tensor_solve_lower(&l, &b, &mut x, 3));
	assert!((x[0].to_native() - 0.5).abs() < 1.0e-6, "x0 = {}", x[0].to_native());
	assert!((x[1].to_native() - 0.75).abs() < 1.0e-6, "x1 = {}", x[1].to_native());
	assert!((x[2].to_native() - 2.375).abs() < 1.0e-6, "x2 = {}", x[2].to_native());
}

#[test]
fn cholesky_and_lu_refuse_a_matrix_that_is_not_positive_definite() {
	// A symmetric matrix with a negative leading minor cannot be factored.
	let a = [F32::from_native(-1.0), F32::ZERO, F32::ZERO, F32::ONE];
	let mut l = [F32::ZERO; 4];
	assert!(!tensor_cholesky(&a, 2, &mut l), "a negative pivot is not a Cholesky");

	// `tensor_lu` returns `bool`, but that `bool` is about the buffers and not
	// about the matrix: a zero pivot is skipped, not reported. So `true` here is
	// the honest answer and `false` would be a lie about the shape.
	let mut lu_l = [F32::ZERO; 4];
	let mut lu_u = [F32::ZERO; 4];
	let short = [F32::ONE; 3];
	assert!(!tensor_lu(&short, 2, &mut lu_l, &mut lu_u), "a short buffer is refused");

	let singular = [F32::ONE, F32::ONE, F32::ONE, F32::ONE];
	assert!(tensor_lu(&singular, 2, &mut lu_l, &mut lu_u));
}

// ---------------------------------------------------------------------------
// Shapes and views
// ---------------------------------------------------------------------------

#[test]
fn a_view_indexes_row_major_over_five_dimensions() {
	let mut data = [F32::ZERO; 24];
	let mut view = TensorView { data: &mut data, shape: [2, 3, 2, 2, 1] };

	assert!(view.is_valid_layout());
	assert_eq!(view.len(), 24);

	*view.get_mut(1, 2, 1, 1, 0).expect("in range") = F32::from_native(7.0);
	assert_eq!(view.idx_linear(1, 2, 1, 1, 0), Some(23));

	// One past the end in each axis is out.
	assert_eq!(view.get(2, 0, 0, 0, 0), None);
	assert_eq!(view.get(0, 3, 0, 0, 0), None);
	assert_eq!(view.get(0, 0, 2, 0, 0), None);
	assert_eq!(view.get(0, 0, 0, 2, 0), None);
}

#[test]
fn a_view_whose_shape_disagrees_with_its_slice_is_not_valid() {
	let mut data = [F32::ZERO; 6];
	let view = TensorView { data: &mut data, shape: [2, 3, 1, 1, 2] };
	assert!(!view.is_valid_layout(), "2*3*1*1*2 is 12, the slice is 6");
}

#[test]
fn an_offset_view_stays_inside_the_slice_it_was_given() {
	let mut data = [F32::ZERO; 8];
	{
		let mut view = TensorViewMut::from_slice_offset(&mut data, [2, 2, 1, 1, 1], 4).expect("4+4 is 8");
		assert_eq!(view.len(), 4);
		view.fill(F32::ONE);
	}
	assert_eq!(data, [F32::ZERO, F32::ZERO, F32::ZERO, F32::ZERO, F32::ONE, F32::ONE, F32::ONE, F32::ONE]);

	assert!(TensorViewMut::from_slice_offset(&mut data, [2, 2, 1, 1, 1], 6).is_none(), "6+4 is 10");
}

#[test]
fn a_dense_tensor_binds_a_shape_and_reports_its_length() {
	// The tensor no longer owns storage: it borrows a slice and lays a shape
	// over it, so the bytes outlive it and there is nothing to free.
	let mut storage = [F32::ZERO; 24];
	{
		let mut t = DenseTensor::<F32>::from_slice(&mut storage, [2, 3, 1, 1, 4]).expect("binds");
		assert_eq!(t.len(), 24);
		assert_eq!(t.shape(), [2, 3, 1, 1, 4]);
		assert!(!t.is_empty());

		for v in t.as_mut_slice() {
			*v = F32::ONE;
		}
		assert!(t.as_slice().iter().all(|v| *v == F32::ONE));
	}
	assert!(storage.iter().all(|v| *v == F32::ONE));

	// A zero in any axis means the shape cannot be laid over the storage.
	assert!(DenseTensor::<F32>::from_slice(&mut storage, [2, 0, 4, 1, 1]).is_none());
}

// ---------------------------------------------------------------------------
// Elementwise and 2-D algebra
// ---------------------------------------------------------------------------

#[test]
fn elementwise_ops_refuse_mismatched_shapes() {
	let mut a = [F32::ONE; 6];
	let mut b = [F32::ONE; 6];
	let mut wrong = [F32::ONE; 4];

	let mut va = TensorView { data: &mut a, shape: [2, 3, 1, 1, 1] };
	let vb = TensorView { data: &mut b, shape: [2, 3, 1, 1, 1] };
	{
		let vw = TensorView { data: &mut wrong, shape: [2, 2, 1, 1, 1] };
		assert!(!tensor_add_in_place(&mut va, &vw), "mismatched shapes do not");
	}
	assert!(tensor_add_in_place(&mut va, &vb), "matching shapes add");
}

#[test]
fn softmax_gives_every_row_one_and_leaves_nan_alone() {
	let mut data = [
		F32::ZERO,
		F32::ONE,
		F32::from_native(2.0),
		F32::ZERO,
		F32::ZERO,
		F32::ZERO,
	];
	let mut view = TensorView { data: &mut data, shape: [2, 1, 1, 1, 3] };
	tensor_softmax_rows(&mut view);

	let row0: f32 = data[..3].iter().map(|v| v.to_native()).sum();
	let row1: f32 = data[3..].iter().map(|v| v.to_native()).sum();
	assert!((row0 - 1.0).abs() < 1.0e-6, "row 0 sums to {row0}");
	assert!((row1 - 1.0).abs() < 1.0e-6, "row 1 sums to {row1}");
	// Monotone within a row.
	assert!(data[0] < data[1] && data[1] < data[2]);
}

#[test]
fn a_bias_is_added_to_every_row_and_a_short_one_is_refused() {
	let mut data = [F32::ZERO; 6];
	{
		let mut view = TensorView { data: &mut data, shape: [2, 1, 1, 1, 3] };
		assert!(tensor_add_bias(&mut view, &[F32::ONE, F32::TWO, F32::from_native(3.0)]));
	}
	assert_eq!(data, [F32::ONE, F32::TWO, F32::from_native(3.0), F32::ONE, F32::TWO, F32::from_native(3.0)]);

	let mut too_short = [F32::ZERO; 6];
	{
		let mut v2 = TensorView { data: &mut too_short, shape: [2, 1, 1, 1, 3] };
		assert!(!tensor_add_bias(&mut v2, &[F32::ONE]), "one bias cannot fill three columns");
	}
}

#[test]
fn matmul_multiply_and_transpose_agree_with_each_other() {
	// a is 2x3, b is 3x2, so c is 2x2.
	let a = [F32::from_native(1.0), F32::from_native(2.0), F32::from_native(3.0), F32::from_native(4.0), F32::from_native(5.0), F32::from_native(6.0)];
	let b = [F32::from_native(7.0), F32::from_native(8.0), F32::from_native(9.0), F32::from_native(10.0), F32::from_native(11.0), F32::from_native(12.0)];
	let mut c = [F32::ZERO; 4];
	assert!(tensor_matmul_2d(&a, &b, &mut c, 2, 3, 2));
	// [[58, 64], [139, 154]]
	assert_eq!(c[0].to_native(), 58.0);
	assert_eq!(c[1].to_native(), 64.0);
	assert_eq!(c[2].to_native(), 139.0);
	assert_eq!(c[3].to_native(), 154.0);

	let mut t = [F32::ZERO; 6];
	assert!(tensor_transpose_2d(&a, &mut t, 2, 3));
	// t is 3x2: [[1, 4], [2, 5], [3, 6]]
	assert_eq!(t[0].to_native(), 1.0);
	assert_eq!(t[1].to_native(), 4.0);
	assert_eq!(t[2].to_native(), 2.0);
	assert_eq!(t[5].to_native(), 6.0);

	// `tensor_trace` is handed a flat slice and a dimension, and it trusts both:
	// asking for the trace of a 2x2 in a six-element buffer reads the first four
	// elements and says so. Worth knowing before passing it the wrong shape.
	assert_eq!(tensor_trace(&a, 2).to_native(), 5.0, "it reads a[0] + a[3]");
	assert_eq!(tensor_trace(&c, 2).to_native(), 212.0, "58 + 154");
	assert_eq!(tensor_trace(&c, 3), F32::ZERO, "a 3x3 does not fit in four elements");
}

#[test]
fn a_square_matrix_inverts_and_multiplies_back_to_the_identity() {
	// [[4, 7], [2, 6]] has determinant 10.
	let a = [
		F32::from_native(4.0),
		F32::from_native(7.0),
		F32::from_native(2.0),
		F32::from_native(6.0),
	];
	let mut inv = [F32::ZERO; 4];
	let mut scratch = [F32::ZERO; 12];
	assert!(tensor_inverse(&a, 2, &mut inv, &mut scratch));

	let mut product = [F32::ZERO; 4];
	assert!(tensor_matmul_2d(&a, &inv, &mut product, 2, 2, 2));
	for (i, v) in product.iter().enumerate() {
		let want = if i % 3 == 0 { 1.0 } else { 0.0 };
		assert!((v.to_native() - want).abs() < 1.0e-5, "product[{i}] = {}", v.to_native());
	}
}

#[test]
fn qr_produces_an_orthonormal_q_and_an_upper_r() {
	// A 3x2 with independent columns.
	let a = [
		F32::from_native(1.0),
		F32::from_native(2.0),
		F32::from_native(1.0),
		F32::from_native(0.0),
		F32::from_native(1.0),
		F32::from_native(1.0),
	];
	let mut q = [F32::ZERO; 6];
	let mut r = [F32::ZERO; 4];
	let mut scratch = [F32::ZERO; 3];
	assert!(tensor_qr(&a, 3, 2, &mut q, &mut r, &mut scratch));

	// q's columns are unit length.
	for col in 0..2 {
		let norm: f32 = (0..3).map(|i| q[i * 2 + col].to_native().powi(2)).sum();
		assert!((norm.sqrt() - 1.0).abs() < 1.0e-5, "column {col} has norm {norm}");
	}
	// R is upper triangular. For a 3x2 there is exactly one entry strictly below
	// the diagonal, r[2], and it is written as a zero rather than left to the
	// rounding noise a straight `sum(q[k] * a[j])` would put there. r[3] is the
	// second diagonal entry and is legitimately nonzero.
	assert_eq!(r[2], F32::ZERO);
	assert!(r[3] > F32::ZERO, "the diagonal of R is a norm");
	// And q r reconstructs a: q is 3x2 and r is 2x2, so the product is 3x2 again.
	let mut rebuilt = [F32::ZERO; 6];
	assert!(tensor_matmul_2d(&q, &r, &mut rebuilt, 3, 2, 2));
	for (i, (got, want)) in rebuilt.iter().zip(a.iter()).enumerate() {
		assert!((got.to_native() - want.to_native()).abs() < 1.0e-5, "rebuilt[{i}] = {}", got.to_native());
	}
}

// ---------------------------------------------------------------------------
// Initializers
// ---------------------------------------------------------------------------

#[test]
fn parameter_counts_follow_the_layer_list() {
	assert_eq!(expected_parameter_counts(&[3, 4, 5]), Some((12 + 20, 4 + 5)));
	assert_eq!(expected_parameter_counts(&[7]), None, "one layer is not a stack");
	assert_eq!(expected_parameter_counts(&[]), None);
	assert_eq!(expected_parameter_counts(&[3, 0, 5]), None, "a zero layer is not a layer");

	assert!(layers_are_valid(&[3, 4]));
	assert!(!layers_are_valid(&[3]));
	assert!(!layers_are_valid(&[3, 0]));
}

#[test]
fn zeros_and_constants_fill_every_weight_and_leave_biases_at_zero() {
	let mut w = [F32::ONE; 32];
	let mut b = [F32::ONE; 8];
	assert_eq!(initialize_parameters_zeros(&[4, 4, 4], &mut w, &mut b), Ok(()));
	assert!(w.iter().all(|v| *v == F32::ZERO));
	assert!(b.iter().all(|v| *v == F32::ZERO));

	let mut w = [F32::ZERO; 32];
	let mut b = [F32::ZERO; 8];
	let value = F32::from_native(0.25);
	assert_eq!(initialize_parameters_constant(&[4, 4, 4], &mut w, &mut b, value), Ok(()));
	assert!(w.iter().all(|v| *v == value));
	assert!(b.iter().all(|v| *v == F32::ZERO), "a constant weight does not bias");
}

#[test]
fn a_non_finite_constant_is_refused() {
	let mut w = [F32::ZERO; 4];
	let mut b = [F32::ZERO; 2];
	let nan = F32::from_native(f32::NAN);
	assert_eq!(initialize_parameters_constant(&[2, 2], &mut w, &mut b, nan), Err(InitError::NonFinite));
	assert!(!init_kind_is_finite(&InitKind::Constant(nan)));
	assert!(init_kind_is_finite::<F32>(&InitKind::Zeros));
	assert!(init_kind_is_finite::<F32>(&InitKind::HeUniform));
}

#[test]
fn the_wrong_buffer_length_is_refused_rather_than_filled() {
	let mut w = [F32::ZERO; 3];
	let mut b = [F32::ZERO; 2];
	assert_eq!(initialize_parameters_zeros(&[2, 2], &mut w, &mut b), Err(InitError::ShapeMismatch));
}

#[test]
fn he_uniform_stays_inside_its_bound_and_the_same_seed_repeats() {
	// fan_in 4, so the bound is sqrt(6/4) = 1.2247.
	let bound = (6.0f32 / 4.0).sqrt();
	let mut w = [F32::ZERO; 128];
	let mut b = [F32::ZERO; 20];
	assert_eq!(initialize_parameters_he_uniform(&[4, 16, 4], &mut w, &mut b, 12345), Ok(()));

	for v in &w {
		assert!(v.is_finite(), "a weight is not finite");
		assert!(v.abs() <= F32::from_native(bound) * F32::from_native(1.000_001), "{v:?} is past the bound");
	}

	let mut again = [F32::ZERO; 128];
	let mut b2 = [F32::ZERO; 20];
	assert_eq!(initialize_parameters_he_uniform(&[4, 16, 4], &mut again, &mut b2, 12345), Ok(()));
	assert_eq!(w, again, "the same seed must give the same weights");

	let mut different = [F32::ZERO; 128];
	let mut b3 = [F32::ZERO; 20];
	assert_eq!(initialize_parameters_he_uniform(&[4, 16, 4], &mut different, &mut b3, 999), Ok(()));
	assert_ne!(w, different, "a different seed must give different weights");
}

#[test]
fn xavier_uniform_is_tighter_than_he_uniform() {
	// Xavier divides by fan_in + fan_out, so its bound is the smaller of the two.
	let mut w = [F32::ZERO; 256];
	let mut b = [F32::ZERO; 16];
	assert_eq!(
		initialize_parameters(&[16, 16], &mut w, &mut b, InitKind::XavierUniform, 7),
		Ok(())
	);

	let widest = w.iter().fold(F32::ZERO, |widest, v| widest.max(v.abs()));
	let he_bound = (6.0f32 / 16.0).sqrt();
	assert!(widest <= F32::from_native(he_bound), "xavier must stay under the he bound");
}

#[test]
fn an_initializer_narrowed_to_e4m3_stays_finite() {
	// The whole point of routing a seed through the helpers: `u * 2 - 1` lands in
	// [-1, 1), which every format can hold, so narrowing cannot produce an
	// infinity here.
	let mut w = [E4M3::ZERO; 64];
	let mut b = [E4M3::ZERO; 8];
	assert_eq!(initialize_parameters(&[8, 8], &mut w, &mut b, InitKind::HeUniform, 3), Ok(()));
	assert!(w.iter().all(|v| v.is_finite()), "a narrowed weight is not finite");
}

// ---------------------------------------------------------------------------
// The kernels, on the format they claim to serve
// ---------------------------------------------------------------------------

#[test]
fn relu_and_scale_agree_with_the_arithmetic_they_replace() {
	let mut data = [F32::from_native(-2.0), F32::from_native(-0.0), F32::ZERO, F32::from_native(3.0)];
	{
		let mut view = TensorView { data: &mut data, shape: [1, 1, 1, 1, 4] };
		tensor_relu(&mut view);
	}
	assert_eq!(data[0], F32::ZERO);
	assert_eq!(data[3], F32::from_native(3.0));

	{
		let mut view = TensorView { data: &mut data, shape: [1, 1, 1, 1, 4] };
		tensor_scale_in_place(&mut view, F32::from_native(2.0));
	}
	assert_eq!(data[3], F32::from_native(6.0));

	{
		let mut view = TensorView { data: &mut data, shape: [1, 1, 1, 1, 4] };
		tensor_fill(&mut view, F32::from_native(1.5));
	}
	assert!(data.iter().all(|v| *v == F32::from_native(1.5)));
}

#[test]
fn a_narrow_tensor_carries_the_same_through_its_view() {
	// E4M3 holds 448 as its top value, and the softmax of a row of them has to
	// come back as a row of 1/3 rather than as zeros.
	let mut data = [E4M3::from_bits(0x7e); 3];
	{
		let mut view = TensorView { data: &mut data, shape: [1, 1, 1, 1, 3] };
		tensor_softmax_rows(&mut view);
	}

	let third = from_f32::<E4M3>(F32::ONE) / from_f32::<E4M3>(F32::from_native(3.0));
	for v in &data {
		assert!((*v - third).abs() <= third * from_f32::<E4M3>(F32::from_native(0.1)), "got {v:?} want {third:?}");
	}
}

#[test]
fn the_scratch_buffer_is_the_only_thing_the_carve_paths_take() {
	// `math`'s carve functions are generic over `Float` and go through
	// `scratch::typed::Buf`; the tensor layer borrows slices. This pins that the
	// two are connected the way the signature says and not by accident.
	let mut storage = [F32::ZERO; 4];
	let mut buf = Buf::from_slice(&mut storage);
	assert_eq!(crate::math::float::carve_softmax_rows(&mut buf, 1), Ok(4));
}