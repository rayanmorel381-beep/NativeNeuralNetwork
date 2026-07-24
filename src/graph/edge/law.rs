use crate::base::math::Float;
use crate::graph::signal::{Mode, Signal};

const WAVE_SPEED: u32 = 1;
const EVENT_THRESHOLD: f32 = 0.5;

pub(crate) fn transport_law<T: Float>(signal: &Signal<'_, T>) -> (u32, bool) {
    match signal.propagation_mode {
        Mode::Dense => (0, true),
        Mode::Sequential => (1, true),
        Mode::Wave => (signal.metadata.distance / WAVE_SPEED, true),
        Mode::Event => (0, signal.confidence >= EVENT_THRESHOLD),
    }
}
