use super::tensor_core::TensorView;
use crate::math::float::Float;

fn checked_product(values: &[usize]) -> Option<usize> {
	values.iter().try_fold(1usize, |product, &value| product.checked_mul(value))
}

fn valid_layout<T: Float>(tensor: &TensorView<'_, T>) -> bool {
	checked_product(&tensor.shape) == Some(tensor.data.len())
}

pub(crate) fn tensor_fill<T: Float>(tensor: &mut TensorView<'_, T>, value: T) {
	for v in tensor.data.iter_mut() {
		*v = value;
	}
}

pub(crate) fn tensor_scale_in_place<T: Float>(tensor: &mut TensorView<'_, T>, scale: T) {
	for v in tensor.data.iter_mut() {
		*v *= scale;
	}
}

pub(crate) fn tensor_add_in_place<T: Float>(dst: &mut TensorView<'_, T>, src: &TensorView<'_, T>) -> bool {
	if dst.shape != src.shape || !valid_layout(dst) || !valid_layout(src) {
		return false;
	}
	for i in 0..dst.data.len() {
		dst.data[i] += src.data[i];
	}
	true
}

pub(crate) fn tensor_relu<T: Float>(tensor: &mut TensorView<'_, T>) {
	for v in tensor.data.iter_mut() {
		if *v < T::ZERO {
			*v = T::ZERO;
		}
	}
}

pub(crate) fn tensor_add_bias<T: Float>(tensor: &mut TensorView<'_, T>, bias: &[T]) -> bool {
	let [n, c, d, h, w] = tensor.shape;
	let rows = match checked_product(&[n, c, d, h]) {
		Some(rows) => rows,
		None => return false,
	};
	if !valid_layout(tensor) || w == 0 || bias.len() < w {
		return false;
	}
	for r in 0..rows {
		let row = &mut tensor.data[r * w..(r + 1) * w];
		for (v, &b) in row.iter_mut().zip(bias.iter()) {
			*v += b;
		}
	}
	true
}

pub(crate) fn tensor_softmax_rows<T: Float>(tensor: &mut TensorView<'_, T>) {
	let [n, c, d, h, w] = tensor.shape;
	let rows = match checked_product(&[n, c, d, h]) {
		Some(rows) => rows,
		None => return,
	};
	if !valid_layout(tensor) || w == 0 {
		return;
	}
	for r in 0..rows {
		let row = &mut tensor.data[r * w..(r + 1) * w];
		let max = row.iter().copied().fold(T::NEG_INF, T::max);
		let mut sum = T::ZERO;
		for v in row.iter_mut() {
			*v = (*v - max).exp();
			sum += *v;
		}
		if sum > T::ZERO {
			for v in row.iter_mut() {
				*v /= sum;
			}
		}
	}
}

pub(crate) fn tensor_matmul_add<T: Float>(
	a: &TensorView<'_, T>,
	b: &TensorView<'_, T>,
	out: &mut TensorView<'_, T>,
	m: usize,
	k: usize,
	n: usize,
) -> bool {
	let Some(a_len) = m.checked_mul(k) else { return false; };
	let Some(b_len) = k.checked_mul(n) else { return false; };
	let Some(out_len) = m.checked_mul(n) else { return false; };
	if !valid_layout(a) || !valid_layout(b) || !valid_layout(out)
		|| a.data.len() < a_len || b.data.len() < b_len || out.data.len() < out_len {
		return false;
	}
	for i in 0..m {
		let ar = &a.data[i * k..i * k + k];
		for j in 0..n {
			let br_col: T = (0..k).map(|p| ar[p] * b.data[p * n + j]).fold(T::ZERO, |sum, value| sum + value);
			out.data[i * n + j] += br_col;
		}
	}
	true
}
