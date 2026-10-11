//! Behaviour tests, kept out of the modules they exercise.
//!
//! No `#[cfg(test)]` module lives next to the code it checks. Everything here
//! is a test of the crate's behaviour, not of its layout, so it sits in its own
//! tree and reaches into `scratch` the way any other caller would.
//!
//! The algebra in `scratch/carve.rs` and the borrow structure in
//! `scratch/arena.rs` and `scratch/slot.rs` are both reversible by reading, which
//! is not a guarantee: a rewind that hands out the same bytes twice, or a split
//! whose parts overlap, is wrong in a way no signature catches. Each test below
//! states one rule of the ownership contract and breaks it if violated.
//!
//! Two of them are regressions for defects found by reading rather than by
//! running: a slot carve that handed out zero bytes, and `get_mut` that refused
//! every slot it was meant to reach.

mod support;

mod activations;
mod frame;
mod math;
mod pool;
mod scratch;
mod tensor;
mod typed;
