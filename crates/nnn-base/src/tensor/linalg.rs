use crate::math::float::Float;

fn checked_square(n: usize) -> Option<usize> { n.checked_mul(n) }

/// The largest magnitude in `values`, or zero when it is empty.
pub(crate) fn max_abs<T: Float>(values: &[T]) -> T {
	values.iter().fold(T::ZERO, |widest, &value| widest.max(value.abs()))
}

/// Whether `value` is too small to divide by, judged against `scale`.
///
/// A pivot cannot be tested against a fixed epsilon. `1e-30` is a perfectly good
/// matrix on its own and a singular one scaled up by `1e30`, and a threshold
/// that is right for `F32` is not right for anything else: as an `E4M3` or an
/// `F16` pattern it rounds to zero, and a comparison against zero never fires,
/// so a singular factorisation would run on and return infinities instead of
/// saying no.
///
/// What makes a value negligible is being small next to the numbers around it,
/// so the test is relative, and `EPSILON` supplies the format's own idea of
/// "small" — which means it moves with the precision instead of being pinned to
/// one format's rounding.
pub(crate) fn is_negligible<T: Float>(value: T, scale: T) -> bool {
	value.abs() <= scale * T::EPSILON
}

pub(crate) fn tensor_trace<T: Float>(a: &[T], n: usize) -> T {
	let Some(nn) = checked_square(n) else { return T::ZERO; };
	if a.len() < nn {
		return T::ZERO;
	}
	let mut sum = T::ZERO;
	for i in 0..n {
		sum += a[i * n + i];
	}
	sum
}

pub(crate) fn tensor_matmul_2d<T: Float>(
	a: &[T],
	b: &[T],
	c: &mut [T],
	m: usize,
	k: usize,
	n: usize,
) -> bool {
	let Some(mk) = m.checked_mul(k) else { return false; };
	let Some(kn) = k.checked_mul(n) else { return false; };
	let Some(mn) = m.checked_mul(n) else { return false; };
	if a.len() < mk || b.len() < kn || c.len() < mn {
		return false;
	}
	for v in c[..mn].iter_mut() {
		*v = T::ZERO;
	}
	for i in 0..m {
		for kk in 0..k {
			let a_ik = a[i * k + kk];
			if a_ik.abs() > T::ZERO {
				for j in 0..n {
					c[i * n + j] += a_ik * b[kk * n + j];
				}
			}
		}
	}
	true
}

pub(crate) fn tensor_transpose_2d<T: Float>(a: &[T], out: &mut [T], m: usize, n: usize) -> bool {
	let Some(mn) = m.checked_mul(n) else { return false; };
	if a.len() < mn || out.len() < mn {
		return false;
	}
	for i in 0..m {
		for j in 0..n {
			out[j * m + i] = a[i * n + j];
		}
	}
	true
}

pub(crate) fn tensor_solve_lower<T: Float>(l: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
	let Some(nn) = checked_square(n) else { return false; };
	if l.len() < nn || b.len() < n || x.len() < n {
		return false;
	}
	for i in 0..n {
		let mut val = b[i];
		for j in 0..i {
			val -= l[i * n + j] * x[j];
		}
		let lii = l[i * n + i];
		if is_negligible(lii, max_abs(&l[..nn])) {
			return false;
		}
		x[i] = val / lii;
	}
	true
}

pub(crate) fn tensor_solve_upper<T: Float>(u: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
	let Some(nn) = checked_square(n) else { return false; };
	if u.len() < nn || b.len() < n || x.len() < n {
		return false;
	}
	for i in (0..n).rev() {
		let mut val = b[i];
		for j in i + 1..n {
			val -= u[i * n + j] * x[j];
		}
		let uii = u[i * n + i];
		if is_negligible(uii, max_abs(&u[..nn])) {
			return false;
		}
		x[i] = val / uii;
	}
	true
}
