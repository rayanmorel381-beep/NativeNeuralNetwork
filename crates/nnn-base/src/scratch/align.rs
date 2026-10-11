//! Address arithmetic, checked.
//!
//! Alignment is the one piece of pointer arithmetic that a memory pillar cannot
//! delegate: a carve is only usable if it starts on a boundary the element type
//! requires, and that boundary is whatever the caller's region happened to be
//! aligned to. Three functions cover the whole need, all `const` so they can be
//! used in constants and evaluated before any byte is touched.
//!
//! - [`is_aligned`] — is this address on a boundary?
//! - [`align_up`] — the next boundary at or after a value.
//! - [`align_down`] — the last boundary at or before a value.
//!
//! # Overflow
//!
//! Rounding up can leave the range of `usize` entirely, so [`align_up`] returns
//! `None` instead of wrapping to a small number that would silently point into
//! the middle of the region. Callers turn that into
//! [`CarveError::Overflow`](crate::scratch::carve::CarveError::Overflow) and the
//! request is rejected.
//!
//! A non power of two alignment is rejected by all three, which is what makes
//! the masking arithmetic below sound.

pub const fn is_aligned(
    address: usize,
    alignment: usize,
) -> bool {
    if alignment == 0 || !alignment.is_power_of_two() {
        return false;
    }

    address & (alignment - 1) == 0
}

pub const fn align_up(
    value: usize,
    alignment: usize,
) -> Option<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }

    let mask = alignment - 1;

    match value.checked_add(mask) {
        Some(sum) => Some(sum & !mask),
        None => None,
    }
}

pub const fn align_down(
    value: usize,
    alignment: usize,
) -> Option<usize> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }

    Some(value & !(alignment - 1))
}

pub const fn align_of_size<T>() -> usize {
    core::mem::align_of::<T>()
}