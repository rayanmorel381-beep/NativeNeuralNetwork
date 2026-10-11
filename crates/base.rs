//! The facade: the one public face of `nnn-base`.
//!
//! Everything below is a layer of the tower — scratch, math, activations,
//! initializers, tensor — and every one of them is crate-private. [`LmlpBase`]
//! is the single name a caller outside the crate holds. It owns the scratch
//! region handed in by the host and one instance of each layer, and the layers
//! reach their storage through that one owner rather than through an allocator
//! of their own.
//!
//! The layers are plain fields, not self-references: a layer that needs scratch
//! is given a carve of the same region at call time, so the whole structure can
//! move without invalidating anything. Operations are exposed over `&mut [T]`
//! slices — the caller gets them from [`LmlpBase::carve`] or from its own
//! storage — and the layers are never named beyond this module.
//!
//! This file only defines [`LmlpBase`] and constructs it. Every operation on
//! the base is an inherent method written in this file, so the facade is
//! self-contained rather than distributed across modules.

use crate::math::float::{BF16, DType, F16, F32, F64, Float, MathError};
use crate::math::helpers::LmlpMath;
use crate::scratch::arena::ScratchMark;
use crate::scratch::carve::CarveError;
use crate::scratch::descriptor::{DescSizes, ScratchDesc};
use crate::scratch::helpers::{LmlpScratch, PoolCarveReport};
use crate::scratch::typed::Buf;
use crate::tensor::helpers::LmlpTensor;
use crate::tensor::tensor_core::TensorView;
use crate::edge::{transport_law, Delivery, Edge, EdgeData, EVENT_THRESHOLD, WAVE_SPEED};
use crate::memory::NodeMemory;
use crate::node::{Node, NodeError};
use crate::propagation::{LmlpSignals, PropagationError};
use crate::routing::RoutingError;
use crate::signal::{Meta, Mode, NodeId, Signal, Tick};

/// The single public entry point of `nnn-base`.
///
/// It borrows the caller's storage for its lifetime: there is no allocation
/// inside, so the bytes must outlive the base and are shared with no one else
/// while it is alive. `T` is the element type every layer below operates on.
pub struct LmlpBase<'a, T> {
	scratch: LmlpScratch<'a>,
	math: LmlpMath<T>,
	activations: crate::activations::LmlpActivations<T>,
	initializers: crate::initializers::LmlpInitializers<T>,
	tensor: LmlpTensor<T>,
}

impl<'a, T: Float> LmlpBase<'a, T> {
	/// Builds the base over `storage`, which becomes the only memory it uses.
	pub fn new(storage: &'a mut [u8]) -> Self {
		Self {
			scratch: LmlpScratch::new(storage),
			math: LmlpMath::new(),
			activations: crate::activations::LmlpActivations::new(),
			initializers: crate::initializers::LmlpInitializers::new(),
			tensor: LmlpTensor::new(),
		}
	}

	// -- memory ----------------------------------------------------------

	/// The total size of the scratch region, in bytes.
	pub fn capacity(&self) -> usize {
		self.scratch.capacity()
	}

	/// How many bytes of the region are currently carved.
	pub fn used(&self) -> usize {
		self.scratch.used()
	}

	/// How many bytes of the region are still free.
	pub fn remaining(&self) -> usize {
		self.scratch.remaining()
	}

	/// Carves `len` values of `T` out of the region.
	///
	/// The returned slice lives for the whole borrowed lifetime of the base, so
	/// it can be kept, reused across the layers, and released with
	/// [`LmlpBase::release`].
	pub fn carve(&mut self, len: usize) -> Result<&'a mut [T], CarveError> {
		self.scratch.carve::<T>(len)
	}

	/// The position a [`LmlpBase::release`] can rewind the region to.
	pub fn mark(&self) -> ScratchMark {
		self.scratch.mark()
	}

	/// Rewinds the region to a [mark](LmlpBase::mark), releasing every carve
	/// made after it.
	///
	/// # Safety
	///
	/// No live slice carved from this region may be touched again. The caller
	/// is the only one who knows which carves are still in use.
	pub unsafe fn release(&mut self, mark: ScratchMark) -> Result<(), CarveError> {
		self.scratch.release(mark)
	}

	/// Carves the eight named regions one layer needs in one call.
	pub fn desc(&mut self, sizes: &DescSizes) -> Result<ScratchDesc<'a, T>, CarveError> {
		self.scratch.desc::<T>(sizes)
	}

	/// A [`DescSizes`] with every region at zero.
	pub fn desc_zeroed(&self) -> DescSizes {
		DescSizes::zeroed()
	}

	/// Total elements an empty scratch descriptor costs.
	pub fn desc_total(&self, sizes: &DescSizes) -> Result<usize, CarveError> {
		sizes.total()
	}

	/// Splits `storage` into the eight named regions of `sizes`.
	pub fn desc_from_sizes(&self, storage: &'a mut [T], sizes: &DescSizes) -> Result<ScratchDesc<'a, T>, CarveError> {
		ScratchDesc::from_sizes(storage, sizes)
	}

	/// Reassembles eight regions into a [`ScratchDesc`], in `state, input,
	/// output, weights, bias, gradient, memory, aux` order.
	pub fn desc_new(&self, regions: [&'a mut [T]; 8]) -> ScratchDesc<'a, T> {
		ScratchDesc::new(regions)
	}

	/// Fills every region of `desc` with `zero`.
	pub fn desc_zero(&self, desc: &mut ScratchDesc<'a, T>, zero: T)
	where
		T: Copy,
	{
		desc.zero(zero);
	}

	/// Carves `len` values of `T` and adopts the `Carve` as a typed [`Buf`].
	///
	/// The scratch region is carved by the memory layer and the region is then
	/// adopted by math's typed path, so this is where the two layers meet:
	/// scratch never imports math, math never owns a region.
	pub fn typed_buf(&mut self, len: usize) -> Result<Buf<'a, T>, MathError> {
		let carve = self.scratch.carve_typed::<T>(len)?;
		crate::math::float::carve_buffer::<T>(carve)
	}

	/// Carves `size` bytes with `alignment` out of the region.
	pub fn carve_bytes(&mut self, size: usize, alignment: usize) -> Result<&'a mut [u8], CarveError> {
		self.scratch.carve_bytes(size, alignment)
	}

	/// Carves `len` values of `T` and returns them as a [`Buf`].
	pub fn buf(&mut self, len: usize) -> Result<Buf<'a, T>, CarveError> {
		self.scratch.buf::<T>(len)
	}

	/// The bytes a whole [`DescSizes`] plan would need, aligned for `T`.
	pub fn footprint(&self, sizes: &DescSizes) -> Result<usize, CarveError> {
		self.scratch.footprint::<T>(sizes)
	}

	/// The footprint a single `T` carve of `len` elements will own.
	pub fn needed(&self, len: usize) -> Result<usize, CarveError> {
		self.scratch.needed::<T>(len)
	}

	/// The aligned footprint a `rows` x `cols` matrix of `T` would own.
	pub fn matrix_footprint(&self, rows: usize, cols: usize) -> Result<usize, CarveError> {
		self.scratch.matrix_footprint::<T>(rows, cols)
	}

	/// The aligned footprint a vector of `len` elements of `T` would own.
	pub fn vector_footprint(&self, len: usize) -> Result<usize, CarveError> {
		self.scratch.vector_footprint::<T>(len)
	}

	/// The aligned footprint the caller's own `slice` would need.
	pub fn slice_footprint(&self, slice: &[T]) -> Result<usize, CarveError> {
		self.scratch.slice_footprint::<T>(slice)
	}

	/// Splits the region into one carved `T` block and reports its three parts:
	/// the bytes skipped for alignment, the bytes after the carve, and the bytes.
	pub fn split_take(&mut self, len: usize) -> Result<(usize, usize, usize), CarveError> {
		self.scratch.split_take::<T>(len)
	}

	/// Carves `len` values of `T`, reports the payload, and returns the carve to
	/// its released state.
	pub fn divest_typed(&mut self, len: usize) -> Result<usize, CarveError> {
		self.scratch.divest_typed::<T>(len)
	}

	/// Describes four split requests without touching any memory.
	pub fn split_scan(&self, at: usize, tail: usize, middle: usize) -> Result<usize, CarveError> {
		self.scratch.split_scan(at, tail, middle)
	}

	/// Parses a byte offset written as hexadecimal (`0x`), octal (`0o`), binary
	/// (`0b`) or decimal text into the `usize` a split request expects.
	///
	/// The four bases of a Rust integer literal are understood, and a single `_`
	/// may separate two digits. An empty string, a bare prefix, an unknown digit,
	/// a misplaced `_`, or a value that overflows a `usize` is an error.
	pub fn parse_offset(text: &str) -> Result<usize, CarveError> {
		crate::scratch::carve::parse_offset(text)
	}

	/// Plans stack, heap and slot regions the way `ArenaLayout` sees them.
	pub fn layout_scan(
		&self,
		stack: usize,
		heap: usize,
		slot_size: usize,
		alignment: usize,
	) -> Result<(usize, usize, usize, usize), CarveError> {
		self.scratch.layout_scan(stack, heap, slot_size, alignment)
	}

	/// Carves `len` elements through a size-class [`Pool`].
	///
	/// # Safety
	///
	/// The region handed to the pool is released before it is returned, so the
	/// caller must not hold any slice that aliases the pool storage.
	pub unsafe fn pool_carve(
		&mut self,
		classes: &'a [usize],
		len: usize,
	) -> Result<PoolCarveReport, CarveError> {
		self.scratch.pool_carve::<T>(classes, len)
	}

	/// Carves and returns a `T` block through a fixed-size slot arena.
	pub fn slot_carve(&mut self, slot_count: usize) -> Result<(usize, usize, usize, usize, usize), CarveError> {
		self.scratch.slot_carve::<T>(slot_count)
	}

	/// Writes `bytes` into a fixed log buffer carved out of the region.
	pub fn log_bytes(&mut self, capacity: usize, bytes: &[u8]) -> Result<(usize, usize), CarveError> {
		self.scratch.log_bytes(capacity, bytes)
	}

	/// Whether the allocation gate is closed: the crate never allocates.
	pub fn gate_closed(&self) -> bool {
		self.scratch.gate_closed()
	}

	/// Asks the gate whether an implicit allocation of `size` bytes would slip
	/// through. With the gate closed this is always `None`.
	///
	/// # Safety
	///
	/// Never call with memory that is still referenced elsewhere.
	pub unsafe fn allocation_test(&self, size: usize, alignment: usize) -> (Option<usize>, bool) {
		self.scratch.allocation_test(size, alignment)
	}

	/// Publishes `data` as a shared slice handle and reads its two ends back.
	///
	/// # Safety
	///
	/// The caller must not touch `data` while this handle is shared.
	pub unsafe fn front_back(&mut self, data: &'a mut [T]) -> (T, T) {
		self.scratch.front_back::<T>(data)
	}

	/// Carves `len` elements inside a frame and releases it.
	///
	/// # Safety
	///
	/// Nothing carved inside the frame may be live when the frame is released.
	pub unsafe fn frame_carve(&mut self, len: usize) -> Result<(ScratchMark, usize), CarveError>
	where
		T: 'a,
	{
		self.scratch.frame_carve::<T>(len)
	}

	/// # Safety
	///
	/// No live carve may be touched again after the rewind.
	pub unsafe fn reuse_marked(&mut self, bytes: usize) -> Result<usize, CarveError> {
		self.scratch.reuse_marked(bytes)
	}

	/// # Safety
	///
	/// The last carve must be dead before the rewind.
	pub unsafe fn reuse_last(&mut self) -> Result<usize, CarveError> {
		self.scratch.reuse_last()
	}

	/// # Safety
	///
	/// Every carve handed out by this base must have been released first.
	pub unsafe fn reuse_reset(&mut self) -> Result<usize, CarveError> {
		self.scratch.reuse_reset()
	}

	/// # Safety
	///
	/// Every carve handed out by this base must have been released first.
	pub unsafe fn reset(&mut self) {
		self.scratch.reset();
	}

	// -- math ------------------------------------------------------------

	/// `x -> |x|` over `data`.
	pub fn abs(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.abs(&mut buffer)
	}

	/// `x -> 1/x` over `data`.
	pub fn recip(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.recip(&mut buffer)
	}

	/// `x -> x * 2^exponent` over `data`, by powers of two built in `T`.
	pub fn ldexp(&self, data: &'a mut [T], exponent: i32) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.ldexp(&mut buffer, exponent)
	}

	/// `x -> e^x` over `data`.
	pub fn exp(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.exp(&mut buffer)
	}

	/// `x -> ln x` over `data`.
	pub fn ln(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.ln(&mut buffer)
	}

	/// `x -> sqrt x` over `data`.
	pub fn sqrt(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.sqrt(&mut buffer)
	}

	/// `x -> sin x` over `data`.
	pub fn sin(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.sin(&mut buffer)
	}

	/// `x -> cos x` over `data`.
	pub fn cos(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.cos(&mut buffer)
	}

	/// `x -> tanh x` over `data`.
	pub fn tanh(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.tanh(&mut buffer)
	}

	/// `x -> max(x, 0)` over `data`.
	pub fn relu(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.relu(&mut buffer)
	}

	/// `x -> 1 / (1 + e^-x)` over `data`.
	pub fn sigmoid(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.sigmoid(&mut buffer)
	}

	/// `x -> round(x)` over `data`, ties away from zero.
	pub fn round(&self, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.round(&mut buffer)
	}

	/// `x -> x^exponent` over `data`.
	pub fn pow(&self, data: &'a mut [T], exponent: T) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.pow(&mut buffer, exponent)
	}

	/// Converts every element of `source` from `T` into `target` as `U`.
	pub fn convert<U: Float>(
		&self,
		source: &'a mut [T],
		target: &'a mut [U],
	) -> Result<usize, MathError> {
		let source_buf = Buf::from_slice(source);
		let mut target_buf = Buf::from_slice(target);
		self.math.convert_into(&source_buf, &mut target_buf)
	}

	/// `x -> floor(x)` over `data` as `F32`. The scratch is not involved, so the
	/// element type of the base does not decide the format here.
	pub fn floor_f32(&self, data: &'a mut [F32]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.floor_f32(&mut buffer)
	}

	/// `x -> floor(x)` over `data` as `F64`.
	pub fn floor_f64(&self, data: &'a mut [F64]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.floor_f64(&mut buffer)
	}

	/// `x -> floor(x)` over `data` as `F16`.
	pub fn floor_f16(&self, data: &'a mut [F16]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.floor_f16(&mut buffer)
	}

	/// `x -> floor(x)` over `data` as `BF16`.
	pub fn floor_bf16(&self, data: &'a mut [BF16]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.floor_bf16(&mut buffer)
	}

/// `x -> 1/x` from `source` into `target`; both hold `T::carve_count`
/// elements and may be disjoint slices.
	pub fn recip_into(
		&self,
		source: &'a mut [T],
		target: &'a mut [T],
	) -> Result<usize, MathError> {
		let source_buf = Buf::from_slice(source);
		let mut target_buf = Buf::from_slice(target);
		self.math.recip_into(&source_buf, &mut target_buf)
	}

	/// Adds `right` into `left` in place, element by element; the two must have
	/// the same length.
	pub fn add_into(&self, left: &'a mut [T], right: &'a mut [T]) -> Result<usize, MathError> {
		let mut left_buf = Buf::from_slice(left);
		let right_buf = Buf::from_slice(right);
		self.math.binary_add(&mut left_buf, &right_buf)
	}

	/// Clamps every element of `data` to `[low, high]`.
	pub fn clamp(&self, data: &'a mut [T], low: T, high: T) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.math.clamp(&mut buffer, low, high)
	}

	/// The byte width, exponent bits, fraction bits and bias the registry
	/// carries for `dtype`, in that order.
	pub fn dtype_info(&self, dtype: DType) -> (usize, u32, u32, i32) {
		crate::math::helpers::dtype_info(dtype)
	}

	/// Runs `round`, `recip` and `tanh` on the value `word` names in `dtype`
	/// and returns, in order, the three result words, the value widened to
	/// `F64` bits, and its `F32` pattern (zero for `F64` itself).
	pub fn dtype_scan(
		&self,
		dtype: DType,
		word: u64,
	) -> (u64, u64, u64, u64, u64) {
		crate::math::helpers::dtype_scan(dtype, word)
	}

	// -- activations -----------------------------------------------------

	/// Applies `kind` to `data`, treating it as one row.
	pub fn activation(&self, kind: crate::activations::ActivationKind, data: &'a mut [T]) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.activations.apply(kind, &mut buffer)
	}

	/// Applies `kind` to `rows` equal rows of `data`; `SoftmaxOutput` needs the
	/// split, the other kinds ignore it.
	pub fn activation_rows(&self, kind: crate::activations::ActivationKind, data: &'a mut [T], rows: usize) -> Result<usize, MathError> {
		let mut buffer = Buf::from_slice(data);
		self.activations.apply_rows(kind, &mut buffer, rows)
	}

	/// Applies `kind` to a single scalar value.
	pub fn apply_one(&self, kind: crate::activations::ActivationKind, value: T) -> T {
		kind.apply(value)
	}

	/// Checks that `kind` left `data` in a finite state.
	pub fn validate(&self, kind: crate::activations::ActivationKind, data: &'a mut [T]) -> Result<usize, MathError> {
		let buffer = Buf::from_slice(data);
		self.activations.validate(kind, &buffer)
	}

	// -- initializers ----------------------------------------------------

	/// Fills `weights` and `biases` for `layers` with `kind`.
	///
	/// `layers` is `[in, hidden..., out]`; the two buffers must hold exactly
	/// what [`expected_parameter_counts`](crate::initializers) predicts.
	pub fn initialize(
		&self,
		layers: &[usize],
		weights: &'a mut [T],
		biases: &'a mut [T],
		kind: crate::initializers::InitKind<T>,
		seed: u64,
	) -> Result<(), crate::initializers::InitError> {
		let mut weights_buf = Buf::from_slice(weights);
		let mut biases_buf = Buf::from_slice(biases);
		self.initializers
			.initialize(layers, &mut weights_buf, &mut biases_buf, kind, seed)
	}

	/// The zero scheme, built the same shape as [`LmlpBase::initialize`].
	pub fn init_zeros(
		&self,
		layers: &[usize],
		weights: &'a mut [T],
		biases: &'a mut [T],
	) -> Result<(), crate::initializers::InitError> {
		let mut weights_buf = Buf::from_slice(weights);
		let mut biases_buf = Buf::from_slice(biases);
		self.initializers
			.initialize_zeros(layers, &mut weights_buf, &mut biases_buf)
	}

	/// The He-uniform scheme, seeded like [`LmlpBase::initialize`].
	pub fn init_he_uniform(
		&self,
		layers: &[usize],
		weights: &'a mut [T],
		biases: &'a mut [T],
		seed: u64,
	) -> Result<(), crate::initializers::InitError> {
		let mut weights_buf = Buf::from_slice(weights);
		let mut biases_buf = Buf::from_slice(biases);
		self.initializers.initialize_he_uniform(
			layers,
			&mut weights_buf,
			&mut biases_buf,
			seed,
		)
	}

	/// The constant-over-weights scheme, biases left at zero.
	pub fn init_constant(
		&self,
		layers: &[usize],
		weights: &'a mut [T],
		biases: &'a mut [T],
		value: T,
	) -> Result<(), crate::initializers::InitError> {
		let mut weights_buf = Buf::from_slice(weights);
		let mut biases_buf = Buf::from_slice(biases);
		self.initializers
			.initialize_constant(layers, &mut weights_buf, &mut biases_buf, value)
	}

	// -- tensor: elementwise ---------------------------------------------

	/// Sets every element of `data` to `value`.
	pub fn fill(&self, shape: [usize; 5], value: T, data: &'a mut [T]) {
		let mut view = TensorView { data, shape };
		self.tensor.fill(&mut view, value);
	}

	/// Scales every element of `data` by `scale`.
	pub fn tensor_scale(&self, shape: [usize; 5], scale: T, data: &'a mut [T]) {
		let mut view = TensorView { data, shape };
		self.tensor.scale(&mut view, scale);
	}

	/// Adds `src` into `dst`, elementwise, for equal shapes.
	pub fn tensor_add(
		&self,
		shape: [usize; 5],
		dst: &'a mut [T],
		src: &'a mut [T],
	) -> bool {
		let mut dst_view = TensorView { data: dst, shape };
		let src_view = TensorView { data: src, shape };
		self.tensor.add_in_place(&mut dst_view, &src_view)
	}

	/// `x -> max(x, 0)` over the whole tensor.
	pub fn tensor_relu(&self, shape: [usize; 5], data: &'a mut [T]) {
		let mut view = TensorView { data, shape };
		self.tensor.relu(&mut view);
	}

	/// Adds a per-row `bias` to `data`, rows of width `shape[4]`.
	pub fn tensor_add_bias(&self, shape: [usize; 5], data: &'a mut [T], bias: &[T]) -> bool {
		let mut view = TensorView { data, shape };
		self.tensor.add_bias(&mut view, bias)
	}

	/// Applies a row-wise softmax to `data`, rows of width `shape[4]`.
	pub fn tensor_softmax(&self, shape: [usize; 5], data: &'a mut [T]) {
		let mut view = TensorView { data, shape };
		self.tensor.softmax_rows(&mut view);
	}

	/// Validates `data` against `shape` and reads the element at `index`;
	/// reports the whole-tensor length and that value.
	pub fn tensor_get(
		&self,
		shape: [usize; 5],
		data: &'a mut [T],
		index: [usize; 5],
	) -> Option<(usize, T)> {
		self.tensor.get_at(shape, data, index)
	}

	/// Writes `value` at `index` in a validated view of `data`; reports the
	/// whole-tensor length.
	pub fn tensor_set(
		&self,
		shape: [usize; 5],
		data: &'a mut [T],
		index: [usize; 5],
		value: T,
	) -> Option<usize> {
		self.tensor.set_at(shape, data, index, value)
	}

	/// The read and write base addresses of a validated view of `data`.
	pub fn tensor_ptrs(&self, shape: [usize; 5], data: &'a mut [T]) -> Option<(usize, usize)> {
		self.tensor.ptrs(shape, data)
	}

	// -- tensor: linear algebra ------------------------------------------

	/// `out += a @ b` for a `m×k` times `k×n` product, with `dims = [m, k, n]`.
	pub fn tensor_matmul_add(
		&self,
		shape: [usize; 5],
		a: &'a mut [T],
		b: &'a mut [T],
		out: &'a mut [T],
		dims: [usize; 3],
	) -> bool {
		let a_view = TensorView { data: a, shape };
		let b_view = TensorView { data: b, shape };
		let mut out_view = TensorView { data: out, shape };
		self.tensor
			.matmul_add(&a_view, &b_view, &mut out_view, dims[0], dims[1], dims[2])
	}

	/// `c = a @ b` for a `m×k` times `k×n` product.
	pub fn matmul(
		&self,
		a: &[T],
		b: &[T],
		c: &mut [T],
		m: usize,
		k: usize,
		n: usize,
	) -> bool {
		self.tensor.matmul_2d(a, b, c, m, k, n)
	}

	/// `out = a^T` for `m×n`.
	pub fn transpose(&self, a: &[T], out: &mut [T], m: usize, n: usize) -> bool {
		self.tensor.transpose_2d(a, out, m, n)
	}

	/// The trace of the `n×n` matrix in `a`.
	pub fn trace(&self, a: &[T], n: usize) -> T {
		self.tensor.trace(a, n)
	}

	/// Solves `L x = b` in place, `L` lower triangular `n×n`.
	pub fn solve_lower(&self, l: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
		self.tensor.solve_lower(l, b, x, n)
	}

	/// Solves `U x = b` in place, `U` upper triangular `n×n`.
	pub fn solve_upper(&self, u: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
		self.tensor.solve_upper(u, b, x, n)
	}

	/// The largest magnitude in `values`, or zero when it is empty.
	pub fn max_abs(&self, values: &[T]) -> T {
		self.tensor.max_abs(values)
	}

	/// Whether `value` is negligible against `scale`, the singularity test.
	pub fn is_negligible(&self, value: T, scale: T) -> bool {
		self.tensor.is_negligible(value, scale)
	}

	// -- tensor: decomposition -------------------------------------------

	/// LU factors of the `n×n` matrix in `a`, without pivoting.
	pub fn lu(&self, a: &[T], n: usize, l: &mut [T], u: &mut [T]) -> bool {
		self.tensor.lu(a, n, l, u)
	}

	/// Cholesky factor of the `n×n` positive definite matrix in `a`.
	pub fn cholesky(&self, a: &[T], n: usize, l: &mut [T]) -> bool {
		self.tensor.cholesky(a, n, l)
	}

	/// QR factors of the `m×n` matrix in `a`.
	pub fn qr(
		&self,
		a: &[T],
		m: usize,
		n: usize,
		q: &mut [T],
		r: &mut [T],
		scratch: &mut [T],
	) -> bool {
		self.tensor.qr(a, m, n, q, r, scratch)
	}

	/// Determinant of the `n×n` matrix in `a`.
	pub fn determinant(&self, a: &[T], n: usize, scratch: &mut [T]) -> Option<T> {
		self.tensor.determinant(a, n, scratch)
	}

	/// Inverse of the `n×n` matrix in `a`.
	pub fn inverse(&self, a: &[T], n: usize, out: &mut [T], scratch: &mut [T]) -> bool {
		self.tensor.inverse(a, n, out, scratch)
	}

	/// Eigenvalues of the `n×n` matrix in `a`, by the unshifted QR iteration.
	pub fn eigenvalues(
		&self,
		a: &[T],
		n: usize,
		max_iter: usize,
		tol: T,
		out: &mut [T],
		scratch: &mut [T],
	) -> bool {
		self.tensor.eigenvalues_qr(a, n, max_iter, tol, out, scratch)
	}

	/// Eigenvalues and eigenvectors of the `n×n` matrix in `a`, by the shifted
	/// QR iteration with accumulation, with `dims = [n, max_iter]`.
	pub fn eigenvectors(
		&self,
		a: &[T],
		dims: [usize; 2],
		tol: T,
		eigenvals: &mut [T],
		eigenvecs: &mut [T],
		scratch: &mut [T],
	) -> bool {
		self.tensor
			.eigenvectors_qr(a, dims, tol, eigenvals, eigenvecs, scratch)
	}

	// -- tensor: storage kinds -------------------------------------------

	/// Binds `shape` to `data` as a dense view, fills it with `value`, and
	/// reports the shape and element count. `None` when `data` does not hold
	/// the shape's product.
	pub fn dense_fill(
		&self,
		shape: [usize; 5],
		data: &'a mut [T],
		value: T,
	) -> Option<([usize; 5], usize)> {
		self.tensor.dense_fill(shape, data, value)
	}

	/// Binds `shape` to a read-only `data` and reports its shape and length.
	pub fn shared_size(&self, shape: [usize; 5], data: &[T]) -> Option<([usize; 5], usize)> {
		self.tensor.shared_size(shape, data)
	}

	/// A zero-initialised `N`-element static tensor bound to `shape`, zeroed
	/// again, reporting the reported shape and live length.
	pub fn static_zeroed<const N: usize>(&self, shape: [usize; 5]) -> Option<([usize; 5], usize)> {
		self.tensor.static_zeroed::<N>(shape)
	}

	/// A validated read-only view of `data`: its length and checksum, or `None`
	/// when the shape and the slice disagree.
	pub fn ro_checksum(&self, shape: [usize; 5], data: &[T]) -> Option<(usize, u64)> {
		self.tensor.ro_checksum(shape, data)
	}

	/// Fills a mutable view of `shape` at the start of `data`.
	pub fn view_mut_fill(
		&self,
		shape: [usize; 5],
		data: &'a mut [T],
		value: T,
	) -> Option<usize> {
		self.tensor.view_mut_fill(shape, data, value)
	}

	/// `view += alpha * src` over a mutable view of `shape` at `offset`.
	pub fn view_mut_axpy(
		&self,
		shape: [usize; 5],
		data: &'a mut [T],
		offset: usize,
		alpha: T,
		src: &[T],
	) -> Option<usize> {
		self.tensor.view_mut_axpy(shape, data, offset, alpha, src)
	}

	/// One SGD step: zero the gradient, accumulate `delta`, then descend on
	/// `data` by `lr`; reports the shape and element count.
	pub fn sgd_step(
		&self,
		shape: [usize; 5],
		data: &'a mut [T],
		grad: &'a mut [T],
		delta: &[T],
		lr: T,
	) -> Option<([usize; 5], usize)> {
		self.tensor.sgd_step(shape, data, grad, delta, lr)
	}

	/// Pushes whole tokens into a ring cache and flattens the live ones into
	/// `out`; reports `(used, is_full, last_token_len)`.
	pub fn cache_push(
		&self,
		data: &'a mut [T],
		capacity: usize,
		num_heads: usize,
		head_dim: usize,
		tokens: &[T],
		out: &mut [T],
	) -> Option<(usize, bool, usize)> {
		self.tensor
			.cache_push(data, capacity, num_heads, head_dim, tokens, out)
	}

	/// Wraps a raw ring-cache region and reports the used token count.
	///
	/// # Safety
	///
	/// `ptr` must point to `capacity * num_heads * head_dim` valid `T`s that
	/// stay live across the call.
	pub unsafe fn cache_from_raw(
		&self,
		ptr: *mut T,
		capacity: usize,
		num_heads: usize,
		head_dim: usize,
	) -> Option<usize> {
		unsafe { self.tensor.cache_from_raw(ptr, capacity, num_heads, head_dim) }
	}

	/// Validates a raw GPU buffer against `shape`; reports `(byte_len,
	/// numel_f32, shape)`.
	pub fn gpu_probe(
		&self,
		ptr: *mut u8,
		byte_len: usize,
		shape: [usize; 5],
	) -> Option<(usize, usize, [usize; 5])> {
		self.tensor.gpu_probe(ptr, byte_len, shape)
	}

	// -- tensor: sparse --------------------------------------------------

	/// `y += A x` for a raw sparse `A`; reports its shape and non-zero count.
	///
	/// # Safety
	///
	/// `csr.0`, `csr.1` and `csr.2` must each hold `csr.3` valid elements.
	pub unsafe fn sparse_spmv(
		&self,
		csr: (*const usize, *const usize, *const T, usize),
		dims: [usize; 2],
		x: &[T],
		y: &mut [T],
	) -> Option<([usize; 2], usize)> {
		unsafe { self.tensor.sparse_spmv(csr, dims, x, y) }
	}

	/// Masks `scores` at the structural zeros of a raw sparse pattern.
	///
	/// # Safety
	///
	/// `csr.0`, `csr.1` and `csr.2` must each hold `csr.3` valid elements.
	pub unsafe fn sparse_mask(
		&self,
		csr: (*const usize, *const usize, *const T, usize),
		dims: [usize; 2],
		scores: &mut [T],
	) -> Option<usize> {
		unsafe { self.tensor.sparse_mask(csr, dims, scores) }
	}

	// -- tensor: lazy evaluation ------------------------------------------

	/// Evaluates a lazy tensor into `out` and reports its element count.
	pub fn lazy_evaluate(
		&self,
		eval_fn: fn(*const u8, &mut [T]),
		opaque: *const u8,
		shape: [usize; 5],
		out: &mut [T],
	) -> usize {
		self.tensor.lazy_evaluate(eval_fn, opaque, shape, out)
	}

	// -- tensor: dequantisation -------------------------------------------

	/// Dequantises one row of a packed int8 tensor; reports its shape and
	/// element count.
	pub fn packed_dequant_row(
		&self,
		q: &[i8],
		shape: [usize; 5],
		row: usize,
		scale: T,
		out: &mut [T],
	) -> Option<([usize; 5], usize)> {
		self.tensor.packed_dequant_row(q, shape, row, scale, out)
	}

	/// Dequantises a group-quantised tensor; reports its shape, element count
	/// and group size.
	pub fn quantized_dequant(
		&self,
		q: &[i8],
		scales: &[T],
		group_size: usize,
		shape: [usize; 5],
		out: &mut [T],
	) -> Option<([usize; 5], usize, usize)> {
		self.tensor
			.quantized_dequant(q, scales, group_size, shape, out)
	}

	// -- protocol: signal ------------------------------------------------

	/// Wraps a carved `payload` into a full-confidence signal, validating it.
	pub fn emit(
		&self,
		payload: Buf<'a, T>,
		timestamp: Tick,
		origin: NodeId,
		mode: Mode,
	) -> Result<Signal<'a, T>, PropagationError> {
		Signal::emitted(payload, timestamp, origin, mode)
	}

	/// The width, in elements, of `signal`'s payload.
	pub fn signal_width(&self, signal: &Signal<'_, T>) -> usize {
		signal.width()
	}

	/// The routing metadata `signal` carries.
	pub fn signal_meta(&self, signal: &Signal<'_, T>) -> Meta {
		signal.metadata()
	}

	/// Every propagation mode, in encoding order.
	pub fn modes(&self) -> [Mode; 4] {
		Mode::ALL
	}

	/// Encodes `mode` as its stable byte.
	pub fn mode_code(&self, mode: Mode) -> u8 {
		mode.as_u8()
	}

	/// Decodes a mode from its stable byte, or `None`.
	pub fn mode_from_code(&self, code: u8) -> Option<Mode> {
		Mode::from_u8(code)
	}

	/// Advances `tick` by `delay` steps, saturating on overflow.
	pub fn advance_tick(&self, tick: Tick, delay: u32) -> Tick {
		tick.advance(delay)
	}

	/// Builds a node identity from its raw value.
	pub fn node_id(&self, raw: u32) -> NodeId {
		NodeId(raw)
	}

	/// Builds a logical time step from its raw value.
	pub fn tick(&self, raw: u64) -> Tick {
		Tick(raw)
	}

	/// The neutral routing metadata: zero phase, zero distance, empty mask.
	pub fn neutral_meta(&self) -> Meta {
		Meta::NEUTRAL
	}

	// -- protocol: edge --------------------------------------------------

	/// An edge from `from` to `to`, owning no weights.
	pub fn edge(&self, from: NodeId, to: NodeId) -> EdgeData {
		EdgeData::new(from, to)
	}

	/// An edge owning the weight slice `[w_offset, w_offset + w_len)`.
	pub fn weighted_edge(&self, from: NodeId, to: NodeId, w_offset: usize, w_len: usize) -> EdgeData {
		EdgeData::with_weights(from, to, w_offset, w_len)
	}

	/// An edge carrying explicit routing metadata.
	pub fn routed_edge(
		&self,
		from: NodeId,
		to: NodeId,
		w_offset: usize,
		w_len: usize,
		metadata: Meta,
	) -> EdgeData {
		EdgeData::with_route(from, to, w_offset, w_len, metadata)
	}

	/// The `(delay, phase, decay, deliver)` law for `signal`'s mode.
	pub fn law(&self, signal: &Signal<'_, T>) -> (u32, f32, f32, bool) {
		transport_law(signal)
	}

	/// The delivery `edge` computes for `signal`.
	pub fn deliver(&self, edge: &EdgeData, signal: &Signal<'_, T>) -> Delivery {
		edge.transport(signal)
	}

	/// The delivery any [`Edge`] computes for `signal`.
	pub fn route<E: Edge<T>>(&self, edge: &E, signal: &Signal<'_, T>) -> Delivery {
		edge.transport(signal)
	}

	/// The weight slice `edge` owns inside the carved `buffer`.
	pub fn edge_weights<'b>(&self, edge: &EdgeData, buffer: &'b Buf<'_, T>) -> Option<&'b [T]> {
		edge.weights(buffer)
	}

	/// Distance, in steps, travelled per step in [`Mode::Wave`](crate::signal::Mode::Wave).
	pub fn wave_speed(&self) -> u32 {
		WAVE_SPEED
	}

	/// The minimum confidence a signal needs to be delivered in
	/// [`Mode::Event`](crate::signal::Mode::Event).
	pub fn event_threshold(&self) -> f32 {
		EVENT_THRESHOLD
	}

	// -- protocol: memory ------------------------------------------------

	/// A node memory with no regions at all.
	pub fn no_memory(&self) -> NodeMemory {
		NodeMemory::NONE
	}

	/// The total number of slots across a node memory's three regions.
	pub fn node_memory_total(&self, memory: &NodeMemory) -> usize {
		memory.total()
	}

	/// Splits a carved `buffer` into `(recurrent, opt_moments, norm_stats)`.
	pub fn split_node_memory<'m>(
		&self,
		memory: &NodeMemory,
		buffer: &'m mut Buf<'_, T>,
	) -> Option<(&'m mut [T], &'m mut [T], &'m mut [T])> {
		memory.split(buffer)
	}

	// -- protocol: node --------------------------------------------------

	/// Folds `inbox` through `node`, producing its next signal.
	pub fn update_node<'s, N: Node<T>>(
		&self,
		node: &mut N,
		inbox: &'s [Signal<'s, T>],
	) -> Result<Signal<'s, T>, NodeError> {
		node.update(inbox)
	}

	/// Feeds a gradient through `node`.
	pub fn learn_node<N: Node<T>>(
		&self,
		node: &mut N,
		gradient: &[Signal<'_, T>],
	) -> Result<(), NodeError> {
		node.learn(gradient)
	}

	/// Advances `node`'s internal time to `tick`.
	pub fn synchronize_node<N: Node<T>>(&self, node: &mut N, tick: Tick) {
		node.synchronize(tick)
	}

	// -- protocol: propagation -------------------------------------------

	/// Wraps a carved `payload` into a signal, validating it first.
	pub fn signal_to_lmlp(
		&self,
		payload: Buf<'a, T>,
		timestamp: Tick,
		origin: NodeId,
		mode: Mode,
		confidence: f32,
		metadata: Meta,
	) -> Result<Signal<'a, T>, PropagationError> {
		crate::propagation::signal_to_lmlp(
			payload, timestamp, origin, mode, confidence, metadata,
		)
	}

	/// Re-validates an existing `signal` in place.
	pub fn validate_signal(&self, signal: &Signal<'_, T>) -> Result<(), PropagationError> {
		crate::propagation::validate_lmlp_signal(signal)
	}

	/// Validates every signal in `signals` as one contiguous run.
	pub fn signals_to_lmlp<'s>(
		&self,
		signals: &'s [Signal<'s, T>],
	) -> Result<LmlpSignals<'s, T>, PropagationError> {
		crate::propagation::signals_to_lmlp(signals)
	}

	/// The signals held by a validated `lmlp` run.
	pub fn lmlp_to_signals<'s>(
		&self,
		lmlp: &'s LmlpSignals<'s, T>,
	) -> &'s [Signal<'s, T>] {
		crate::propagation::lmlp_to_signals(lmlp)
	}

	/// Rewrites every signal's mode in place and reports how many were rewritten.
	pub fn reinterpret_signals(
		&self,
		signals: &mut [Signal<'_, T>],
		mode: Mode,
	) -> Result<usize, PropagationError> {
		crate::propagation::reinterpret_lmlp_signals(signals, mode)
	}

	/// How many signals a validated `lmlp` run holds.
	pub fn signal_count(&self, lmlp: &LmlpSignals<'_, T>) -> usize {
		lmlp.len()
	}

	/// Whether a validated `lmlp` run is empty.
	pub fn signals_empty(&self, lmlp: &LmlpSignals<'_, T>) -> bool {
		lmlp.is_empty()
	}

	// -- protocol: routing -----------------------------------------------

	/// Folds an inbox into the carved `out` as `out += Σ w_i · decay_i · payload_i`,
	/// over the signals whose matching `deliver` flag is set. Reports the width.
	pub fn aggregate_inbox(
		&self,
		signals: &[Signal<'_, T>],
		weights: &[T],
		decays: &[T],
		deliver: &[bool],
		out: &mut Buf<'_, T>,
	) -> Result<usize, RoutingError> {
		crate::routing::aggregate_inbox(signals, weights, decays, deliver, out)
	}
}