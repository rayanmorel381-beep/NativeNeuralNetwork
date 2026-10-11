use crate::math::float::Float;
use crate::scratch::typed::Buf;
use crate::signal::{Meta, NodeId, Signal};

use super::{transport_law, Delivery};

/// A routing edge: a source, a destination, the weight slice it owns, and
/// optional routing metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeData {
    pub from: NodeId,
    pub to: NodeId,
    pub w_offset: usize,
    pub w_len: usize,
    pub metadata: Meta,
}

impl EdgeData {
    /// An edge with no weights and neutral metadata.
    pub fn new(from: NodeId, to: NodeId) -> Self {
        EdgeData {
            from,
            to,
            w_offset: 0,
            w_len: 0,
            metadata: Meta::NEUTRAL,
        }
    }

    /// An edge owning the weight slice `[w_offset, w_offset + w_len)`.
    pub fn with_weights(from: NodeId, to: NodeId, w_offset: usize, w_len: usize) -> Self {
        EdgeData {
            from,
            to,
            w_offset,
            w_len,
            metadata: Meta::NEUTRAL,
        }
    }

    /// An edge with explicit routing metadata.
    pub fn with_route(
        from: NodeId,
        to: NodeId,
        w_offset: usize,
        w_len: usize,
        metadata: Meta,
    ) -> Self {
        Self {
            from,
            to,
            w_offset,
            w_len,
            metadata,
        }
    }

    /// The delivery this edge computes for `signal`.
    pub fn transport<T: Float>(&self, signal: &Signal<'_, T>) -> Delivery {
        let (delay, phase, decay, deliver) = transport_law(signal);
        Delivery {
            delay,
            phase,
            decay,
            target: self.to,
            deliver,
        }
    }

    /// The edge's weight slice inside a carved `buffer`.
    ///
    /// Returns `None` when the slice runs past the region.
    pub fn weights<'b, 'a, T: Float>(&self, buffer: &'b Buf<'a, T>) -> Option<&'b [T]> {
        let weights = buffer.as_slice();
        let end = self.w_offset.checked_add(self.w_len)?;
        if end > weights.len() {
            return None;
        }
        Some(&weights[self.w_offset..end])
    }
}

/// Anything that can route a signal.
pub trait Edge<T: Float> {
    /// Computes the delivery for `signal`.
    fn transport(&self, signal: &Signal<'_, T>) -> Delivery;
}

impl<T: Float> Edge<T> for EdgeData {
    fn transport(&self, signal: &Signal<'_, T>) -> Delivery {
        EdgeData::transport(self, signal)
    }
}
