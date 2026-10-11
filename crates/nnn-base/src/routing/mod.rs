//! Inbox aggregation: the pure arithmetic an edge set applies before a node
//! reads its inbox.
//!
//! The accumulator is a [`Buf`] — a carved region of `scratch` — so the module
//! never allocates and never introduces its own memory.

use crate::math::float::Float;
use crate::scratch::typed::Buf;
use crate::signal::Signal;

/// Why an inbox could not be aggregated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingError {
    EmptyInbox,
    WidthMismatch,
    BufferTooSmall,
}

/// Folds an inbox into the carved `out` as `out += Σ w_i · decay_i · payload_i`,
/// over the signals whose matching `deliver` flag is set. Reports the width.
pub fn aggregate_inbox<T: Float>(
    signals: &[Signal<'_, T>],
    weights: &[T],
    decays: &[T],
    deliver: &[bool],
    out: &mut Buf<'_, T>,
) -> Result<usize, RoutingError> {
    let count = signals.len();
    if count == 0 {
        return Err(RoutingError::EmptyInbox);
    }
    if weights.len() != count || decays.len() != count || deliver.len() != count {
        return Err(RoutingError::WidthMismatch);
    }

    let width = signals[0].width();
    if width == 0 {
        return Err(RoutingError::EmptyInbox);
    }
    if out.len() < width {
        return Err(RoutingError::BufferTooSmall);
    }
    if signals.iter().any(|signal| signal.width() != width) {
        return Err(RoutingError::WidthMismatch);
    }

    let accumulator = out.as_mut_slice();
    for (index, signal) in signals.iter().enumerate() {
        if !deliver[index] {
            continue;
        }
        let scale = weights[index] * decays[index];
        for (target, value) in accumulator.iter_mut().zip(signal.data().iter()) {
            *target += scale * *value;
        }
    }

    Ok(width)
}
