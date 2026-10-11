use crate::math::float::Float;

/// A shared, borrowed view of tensor storage.
///
/// It used to be a reference-counted handle over an allocation. The crate's
/// ownership rule is one owner per region, and every region is owned by the
/// caller's `scratch`, so sharing is expressed as another borrow rather than as
/// a count kept in a header. No `Clone`, no `Drop`, no atomics.
pub(crate) struct SharedTensor<'a, T> {
	data: &'a [T],
	pub(crate) shape: [usize; 5],
}

impl<'a, T: Float> SharedTensor<'a, T> {
	/// Binds `shape` to `data`, which must hold exactly the shape's product.
	pub(crate) fn from_slice(
		data: &'a [T],
		shape: [usize; 5],
	) -> Option<Self> {
		let len = shape.iter().fold(1usize, |a, &d| a.saturating_mul(d));

		if len == 0 || data.len() != len {
			return None;
		}

		Some(Self { data, shape })
	}

	pub(crate) fn as_slice(&self) -> &[T] {
		self.data
	}

	pub(crate) fn len(&self) -> usize {
		self.data.len()
	}

	pub(crate) fn shape(&self) -> [usize; 5] {
		self.shape
	}

	pub(crate) fn is_empty(&self) -> bool {
		self.data.is_empty()
	}
}
