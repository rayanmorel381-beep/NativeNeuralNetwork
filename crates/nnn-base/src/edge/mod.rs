//! Edges: they read a signal's mode and decide how (and whether) it is
//! transported to a destination.

pub(crate) mod delivery;
pub(crate) mod edge_data;
pub(crate) mod law;

pub use delivery::Delivery;
pub use edge_data::{Edge, EdgeData};
pub use law::{transport_law, EVENT_THRESHOLD, WAVE_SPEED};
