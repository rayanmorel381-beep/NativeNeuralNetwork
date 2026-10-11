//! Scratch memory pillar for the whole workspace.
//!
//! This module owns the vocabulary and the rules for temporary memory. Nothing
//! here talks to the CPU, allocates implicitly, or reaches into another crate:
//! it is plain safe Rust over a buffer the caller supplies.
//!
//! # The carve machine
//!
//! [`carve`] is the centre. A [`carve::Carve`] moves through three states —
//! planned, carved, released — and describes its intent through
//! [`carve::CarveKind`], which has four families:
//!
//! - `Split` cuts a region out of a larger one at a given offset.
//! - `From` starts from a size or a type instead of a position.
//! - `Reuse` reclaims memory already handed out, the responsibility of
//!   [`arena::Scratch::reuse`].
//! - `Region` binds the carve to a backing region: stack, heap, slot or pool.
//!
//! # The backends
//!
//! Four types implement that machine with different reclamation strategies:
//!
//! - [`arena::Scratch`] — a linear cursor. Forwards when carving, rewinds
//!   through [`carve::ReuseCarve`] when reclaiming.
//! - [`pool`] — carves grouped by size class, tracked through `PoolMark`.
//! - [`slot`] — fixed-size slots handed out in order, released last in first out.
//! - [`frame`] — a marked region that groups a batch of carves and rolls all of
//!   them back together.
//!
//! # Ownership rule
//!
//! One region has at most one owner. A carve owns only the bytes it was cut,
//! never a prefix or a suffix, so two carves can never overlap. Reconstruction
//! of a slice from a raw pointer happens in exactly five places — [`typed`],
//! [`arena`], [`slot`], [`pool`] and [`frame`] — and each one documents what
//! the caller must guarantee.
//!
//! # Typed access
//!
//! [`typed::Buf`] is the only route from bytes to typed values. `math/` depends
//! on it exclusively: no operation there touches a raw pointer, so none of them
//! can violate the ownership rule above.

pub(crate) mod arena;
pub(crate) mod align;
pub(crate) mod bytes;
pub(crate) mod typed;
pub(crate) mod pool;
pub(crate) mod frame;
pub(crate) mod layout;
pub(crate) mod slot;
pub(crate) mod descriptor;
pub(crate) mod mut_ptr;
pub(crate) mod logbuf;
pub(crate) mod galloc;
pub(crate) mod carve;
pub(crate) mod helpers;

pub use self::align::*;
pub use self::arena::*;
pub use self::bytes::*;
pub use self::carve::*;
pub use self::descriptor::*;
pub use self::frame::*;
pub use self::galloc::*;
pub use self::helpers::*;
pub use self::layout::*;
pub use self::logbuf::*;
pub use self::mut_ptr::*;
pub use self::pool::*;
pub use self::slot::*;
pub use self::typed::*;