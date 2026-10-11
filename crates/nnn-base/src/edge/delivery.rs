use crate::signal::NodeId;

/// What an edge decided to do with a signal: when it arrives, its phase and
/// decay, where it is headed, and whether it is delivered at all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Delivery {
    pub delay: u32,
    pub phase: f32,
    pub decay: f32,
    pub target: NodeId,
    pub deliver: bool,
}
