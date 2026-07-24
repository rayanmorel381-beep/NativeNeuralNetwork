use crate::base::math::Float;
use crate::graph::signal::{Meta, Mode, NodeId, Signal};
use super::{transport_law, Delivery};

#[derive(Clone, Copy)]
pub(crate) struct EdgeData {
    pub from: NodeId,
    pub to: NodeId,
    pub mode: Mode,
    pub meta: Meta,
}

impl EdgeData {
    pub(crate) fn new(from: NodeId, to: NodeId) -> Self {
        EdgeData {
            from,
            to,
            mode: Mode::Dense,
            meta: Meta::NEUTRAL,
        }
    }

    pub(crate) fn with_mode(from: NodeId, to: NodeId, mode: Mode, meta: Meta) -> Self {
        EdgeData {
            from,
            to,
            mode,
            meta,
        }
    }

    pub(crate) fn transport<T: Float>(&self, signal: &Signal<'_, T>) -> Delivery {
        let forward = self.to.0 > self.from.0;
        let (delay, mode_allows) = transport_law(signal);
        Delivery {
            delay,
            deliver: forward && mode_allows,
        }
    }
}
