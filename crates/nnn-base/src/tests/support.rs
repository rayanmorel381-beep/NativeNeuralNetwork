//! Shared fixtures for the behaviour tests.
//!
//! This module holds no tests itself. What lives here are the buffers the tests
//! need so that a failure means the carve logic is wrong, rather than that the
//! stack happened to land on a lucky address.

/// A buffer whose address satisfies the alignment most carves ask for.
///
/// A bare `[u8; N]` is only ever 1-aligned, which would make every alignment
/// test pass or fail for a reason that has nothing to do with the carve logic.
///
/// A subslice taken at a non-multiple offset is still usable for the negative
/// alignment cases: `&mut raw.0[1..]` is deliberately misaligned.
#[repr(align(8))]
pub(crate) struct Aligned<const N: usize>(pub(crate) [u8; N]);

/// Size classes for a [`Pool`](crate::scratch::pool::Pool), ascending.
pub(crate) const CLASSES: &[usize] = &[16, 32, 64, 128];
