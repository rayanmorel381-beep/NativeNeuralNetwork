//! The tensor layer, the fifth tier of the tower.
//!
//! The elementwise, linear-algebra and decomposition routines below it are free
//! functions over slices; this layer is the one object `base.rs` holds so a
//! caller reaches them through a single stable name. It keeps no state of its
//! own — every buffer it works on belongs to the caller's scratch — so the only
//! thing it carries is the element type.

use super::cache_tensor::CacheTensor;
use super::decompose;
use super::dense::DenseTensor;
use super::elementwise;
use super::gpu::GpuTensor;
use super::lazy::LazyTensor;
use super::linalg;
use super::packed::PackedTensor;
use super::quantized::QuantizedTensor;
use super::readonly::TensorViewRo;
use super::shared::SharedTensor;
use super::sparse::SparseTensor;
use super::static_tensor::StaticTensor;
use super::tensor_core::TensorView;
use super::training::TrainingTensor;
use super::view_mut::TensorViewMut;
use crate::math::float::Float;
use core::marker::PhantomData;

/// The tensor layer, held by [`LmlpBase`].
///
/// [`LmlpBase`]: crate::base::LmlpBase
pub(crate) struct LmlpTensor<T> {
	_marker: PhantomData<T>,
}

impl<T: Float> LmlpTensor<T> {
	pub(crate) const fn new() -> Self {
		Self {
			_marker: PhantomData,
		}
	}

	// -- elementwise -----------------------------------------------------

	pub(crate) fn fill(&self, view: &mut TensorView<'_, T>, value: T) {
		elementwise::tensor_fill(view, value);
	}

	pub(crate) fn scale(&self, view: &mut TensorView<'_, T>, scale: T) {
		elementwise::tensor_scale_in_place(view, scale);
	}

	pub(crate) fn add_in_place(
		&self,
		dst: &mut TensorView<'_, T>,
		src: &TensorView<'_, T>,
	) -> bool {
		elementwise::tensor_add_in_place(dst, src)
	}

	pub(crate) fn relu(&self, view: &mut TensorView<'_, T>) {
		elementwise::tensor_relu(view);
	}

	pub(crate) fn add_bias(&self, view: &mut TensorView<'_, T>, bias: &[T]) -> bool {
		elementwise::tensor_add_bias(view, bias)
	}

	pub(crate) fn softmax_rows(&self, view: &mut TensorView<'_, T>) {
		elementwise::tensor_softmax_rows(view);
	}

	pub(crate) fn matmul_add(
		&self,
		a: &TensorView<'_, T>,
		b: &TensorView<'_, T>,
		out: &mut TensorView<'_, T>,
		m: usize,
		k: usize,
		n: usize,
	) -> bool {
		elementwise::tensor_matmul_add(a, b, out, m, k, n)
	}

	// -- linear algebra --------------------------------------------------

	pub(crate) fn matmul_2d(
		&self,
		a: &[T],
		b: &[T],
		c: &mut [T],
		m: usize,
		k: usize,
		n: usize,
	) -> bool {
		linalg::tensor_matmul_2d(a, b, c, m, k, n)
	}

	pub(crate) fn transpose_2d(&self, a: &[T], out: &mut [T], m: usize, n: usize) -> bool {
		linalg::tensor_transpose_2d(a, out, m, n)
	}

	pub(crate) fn trace(&self, a: &[T], n: usize) -> T {
		linalg::tensor_trace(a, n)
	}

	pub(crate) fn solve_lower(&self, l: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
		linalg::tensor_solve_lower(l, b, x, n)
	}

	pub(crate) fn solve_upper(&self, u: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
		linalg::tensor_solve_upper(u, b, x, n)
	}

	pub(crate) fn max_abs(&self, values: &[T]) -> T {
		linalg::max_abs(values)
	}

	pub(crate) fn is_negligible(&self, value: T, scale: T) -> bool {
		linalg::is_negligible(value, scale)
	}

	// -- decomposition ---------------------------------------------------

	pub(crate) fn lu(&self, a: &[T], n: usize, l: &mut [T], u: &mut [T]) -> bool {
		decompose::tensor_lu(a, n, l, u)
	}

	pub(crate) fn cholesky(&self, a: &[T], n: usize, l: &mut [T]) -> bool {
		decompose::tensor_cholesky(a, n, l)
	}

	pub(crate) fn qr(
		&self,
		a: &[T],
		m: usize,
		n: usize,
		q: &mut [T],
		r: &mut [T],
		scratch: &mut [T],
	) -> bool {
		decompose::tensor_qr(a, m, n, q, r, scratch)
	}

	pub(crate) fn determinant(&self, a: &[T], n: usize, scratch: &mut [T]) -> Option<T> {
		decompose::tensor_determinant(a, n, scratch)
	}

	pub(crate) fn inverse(&self, a: &[T], n: usize, out: &mut [T], scratch: &mut [T]) -> bool {
		decompose::tensor_inverse(a, n, out, scratch)
	}

	pub(crate) fn eigenvalues_qr(
		&self,
		a: &[T],
		n: usize,
		max_iter: usize,
		tol: T,
		out: &mut [T],
		scratch: &mut [T],
	) -> bool {
		decompose::tensor_eigenvalues_qr(a, n, max_iter, tol, out, scratch)
	}

	pub(crate) fn eigenvectors_qr(
		&self,
		a: &[T],
		dims: [usize; 2],
		tol: T,
		eigenvals: &mut [T],
		eigenvecs: &mut [T],
		scratch: &mut [T],
	) -> bool {
		decompose::tensor_eigenvectors_qr(
			a,
			dims[0],
			dims[1],
			tol,
			eigenvals,
			eigenvecs,
			scratch,
		)
	}

	// -- storage kinds ---------------------------------------------------

	pub(crate) fn get_at(
		&self,
		shape: [usize; 5],
		data: &mut [T],
		index: [usize; 5],
	) -> Option<(usize, T)> {
		let view = TensorView { data, shape };
		if view.is_empty() || !view.is_valid_layout() {
			return None;
		}
		let [n, c, d, h, w] = index;
		let len = view.len();
		let value = view.get(n, c, d, h, w)?;
		Some((len, value))
	}

	pub(crate) fn set_at(
		&self,
		shape: [usize; 5],
		data: &mut [T],
		index: [usize; 5],
		value: T,
	) -> Option<usize> {
		let mut view = TensorView { data, shape };
		if !view.is_valid_layout() {
			return None;
		}
		let [n, c, d, h, w] = index;
		*view.get_mut(n, c, d, h, w)? = value;
		Some(view.len())
	}

	pub(crate) fn ptrs(&self, shape: [usize; 5], data: &mut [T]) -> Option<(usize, usize)> {
		let mut view = TensorView { data, shape };
		if !view.is_valid_layout() {
			return None;
		}
		let read = view.as_ptr() as usize;
		let write = view.as_mut_ptr() as usize;
		Some((read, write))
	}

	pub(crate) fn dense_fill(
		&self,
		shape: [usize; 5],
		data: &mut [T],
		value: T,
	) -> Option<([usize; 5], usize)> {
		let mut view = DenseTensor::from_slice(data, shape)?;
		if view.is_empty() {
			return None;
		}
		let len = view.len();
		for element in view.as_mut_slice() {
			*element = value;
		}
		let slice_len = view.as_slice().len();
		debug_assert_eq!(slice_len, len);
		Some((view.shape(), len))
	}

	pub(crate) fn shared_size(
		&self,
		shape: [usize; 5],
		data: &[T],
	) -> Option<([usize; 5], usize)> {
		let view = SharedTensor::from_slice(data, shape)?;
		if view.is_empty() {
			return None;
		}
		let len = view.as_slice().len();
		let reported = view.len();
		debug_assert_eq!(reported, len);
		Some((view.shape(), len))
	}

	pub(crate) fn static_zeroed<const N: usize>(
		&self,
		shape: [usize; 5],
	) -> Option<([usize; 5], usize)> {
		let mut tensor = StaticTensor::<T, N>::zeroed(shape)?;
		tensor.fill_zero();
		let len = tensor.as_slice().len();
		for element in tensor.as_mut_slice() {
			*element = T::ZERO;
		}
		Some((tensor.shape, len))
	}

	pub(crate) fn ro_checksum(&self, shape: [usize; 5], data: &[T]) -> Option<(usize, u64)> {
		let view = TensorViewRo::from_slice(data, shape);
		if view.is_empty() || !view.is_valid_layout() {
			return None;
		}
		Some((view.len(), view.checksum()))
	}

	pub(crate) fn view_mut_fill(
		&self,
		shape: [usize; 5],
		data: &mut [T],
		value: T,
	) -> Option<usize> {
		let mut view = TensorViewMut::from_slice(data, shape)?;
		if view.is_empty() {
			return None;
		}
		view.fill(value);
		Some(view.len())
	}

	pub(crate) fn view_mut_axpy(
		&self,
		shape: [usize; 5],
		data: &mut [T],
		offset: usize,
		alpha: T,
		src: &[T],
	) -> Option<usize> {
		let mut view = TensorViewMut::from_slice_offset(data, shape, offset)?;
		if view.is_empty() {
			return None;
		}
		view.axpy(alpha, src);
		Some(view.len())
	}

	pub(crate) fn sgd_step(
		&self,
		shape: [usize; 5],
		data: &mut [T],
		grad: &mut [T],
		delta: &[T],
		lr: T,
	) -> Option<([usize; 5], usize)> {
		let mut tensor = TrainingTensor::from_slices(data, grad, shape)?;
		tensor.zero_grad();
		tensor.accumulate_grad(delta);
		tensor.apply_grad(lr);
		let len = tensor.data().len();
		let _ = tensor.data_mut().len();
		let grads = tensor.gradients().len();
		let count = tensor.numel();
		debug_assert_eq!(grads, count);
		Some((tensor.shape(), len))
	}

	pub(crate) fn cache_push(
		&self,
		data: &mut [T],
		capacity: usize,
		num_heads: usize,
		head_dim: usize,
		tokens: &[T],
		out: &mut [T],
	) -> Option<(usize, bool, usize)> {
		let mut cache = CacheTensor::from_slice(data, capacity, num_heads, head_dim)?;
		let stride = cache.stride();
		for chunk in tokens.chunks(stride) {
			cache.push(chunk);
		}
		let used = cache.used_tokens();
		let full = cache.is_full();
		let last = cache.token(used.saturating_sub(1)).map_or(0, <[T]>::len);
		cache.copy_tokens_into(out)?;
		Some((used, full, last))
	}

	/// # Safety
	///
	/// `ptr` must point to `capacity * num_heads * head_dim` valid `T`s that
	/// stay live across the call.
	pub(crate) unsafe fn cache_from_raw(
		&self,
		ptr: *mut T,
		capacity: usize,
		num_heads: usize,
		head_dim: usize,
	) -> Option<usize> {
		let cache = unsafe { CacheTensor::from_raw(ptr, capacity, num_heads, head_dim) }?;
		Some(cache.used_tokens())
	}

	pub(crate) fn gpu_probe(
		&self,
		ptr: *mut u8,
		byte_len: usize,
		shape: [usize; 5],
	) -> Option<(usize, usize, [usize; 5])> {
		let mut tensor = GpuTensor::from_raw(ptr, byte_len, shape)?;
		let bytes = tensor.as_bytes().len();
		debug_assert_eq!(bytes, tensor.byte_len());
		let _ = tensor.as_bytes_mut();
		Some((tensor.byte_len(), tensor.numel_f32(), tensor.shape()))
	}

	/// # Safety
	///
	/// `csr.0`, `csr.1` and `csr.2` must each hold `csr.3` valid elements.
	pub(crate) unsafe fn sparse_spmv(
		&self,
		csr: (*const usize, *const usize, *const T, usize),
		dims: [usize; 2],
		x: &[T],
		y: &mut [T],
	) -> Option<([usize; 2], usize)> {
		let (row_indices, col_indices, values, nnz) = csr;
		let tensor =
			SparseTensor::from_raw(row_indices, col_indices, values, nnz, dims[0], dims[1])?;
		tensor.spmv(x, y);
		Some((tensor.shape(), tensor.nnz()))
	}

	/// # Safety
	///
	/// `csr.0`, `csr.1` and `csr.2` must each hold `csr.3` valid elements.
	pub(crate) unsafe fn sparse_mask(
		&self,
		csr: (*const usize, *const usize, *const T, usize),
		dims: [usize; 2],
		scores: &mut [T],
	) -> Option<usize> {
		let (row_indices, col_indices, values, nnz) = csr;
		let tensor =
			SparseTensor::from_raw(row_indices, col_indices, values, nnz, dims[0], dims[1])?;
		tensor.apply_mask(scores);
		Some(tensor.nnz())
	}

	pub(crate) fn lazy_evaluate(
		&self,
		eval_fn: fn(*const u8, &mut [T]),
		opaque: *const u8,
		shape: [usize; 5],
		out: &mut [T],
	) -> usize {
		let tensor = LazyTensor::new(eval_fn, opaque, shape);
		tensor.evaluate(out);
		tensor.numel()
	}

	pub(crate) fn packed_dequant_row(
		&self,
		q: &[i8],
		shape: [usize; 5],
		row: usize,
		scale: T,
		out: &mut [T],
	) -> Option<([usize; 5], usize)> {
		let tensor = PackedTensor::from_raw(q, shape)?;
		tensor.dequant_row(row, scale, out);
		Some((tensor.shape(), tensor.numel()))
	}

	pub(crate) fn quantized_dequant(
		&self,
		q: &[i8],
		scales: &[T],
		group_size: usize,
		shape: [usize; 5],
		out: &mut [T],
	) -> Option<([usize; 5], usize, usize)> {
		let tensor = QuantizedTensor::from_qweight(q, scales, group_size, shape)?;
		tensor.dequant_into(out);
		Some((tensor.shape(), tensor.numel(), tensor.group_size()))
	}
}
