use crate::base::math::Float;
use super::{Meta, Mode, NodeId, Tick};

pub(crate) struct Signal<'a, T: Float> {
    pub payload: &'a [T],
    pub timestamp: Tick,
    pub confidence: f32,
    pub origin: NodeId,
    pub propagation_mode: Mode,
    pub metadata: Meta,
}

impl<'a, T: Float> Signal<'a, T> {
    pub(crate) fn emitted(
        payload: &'a [T],
        timestamp: Tick,
        origin: NodeId,
        propagation_mode: Mode,
    ) -> Self {
        Signal {
            payload,
            timestamp,
            confidence: 1.0,
            origin,
            propagation_mode,
            metadata: Meta::NEUTRAL,
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.payload.len()
    }
}
