use crate::math::float::Float;
use crate::signal::{Mode, Signal};

/// Distance travelled per step in [`Mode::Wave`].
pub const WAVE_SPEED: u32 = 1;

/// Minimum confidence a signal needs to be delivered in [`Mode::Event`], on top
/// of a non-zero [`Meta::event_mask`](crate::signal::Meta::event_mask).
pub const EVENT_THRESHOLD: f32 = 0.5;

/// The `(delay, phase, decay, deliver)` law for the signal's mode.
pub fn transport_law<T: Float>(signal: &Signal<'_, T>) -> (u32, f32, f32, bool) {
    match signal.propagation_mode() {
        Mode::Dense => (0, 0.0, 1.0, true),
        Mode::Sequential => (1, 0.0, 1.0, true),
        Mode::Wave => (
            signal.metadata().distance / WAVE_SPEED,
            signal.metadata().phase,
            signal.confidence(),
            true,
        ),
        Mode::Event => (
            0,
            signal.metadata().phase,
            1.0,
            signal.metadata().event_mask != 0 && signal.confidence() >= EVENT_THRESHOLD,
        ),
    }
}
