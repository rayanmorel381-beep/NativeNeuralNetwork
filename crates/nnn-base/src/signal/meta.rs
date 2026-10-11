/// Optional per-signal routing hints read by an edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meta {
    pub phase: f32,
    pub distance: u32,
    /// The event types the signal carries, as a bit mask. [`Mode::Event`]
    /// delivers the signal only when this is non-zero.
    ///
    /// [`Mode::Event`]: crate::signal::Mode::Event
    pub event_mask: u32,
}

impl Meta {
    /// No hint: zero phase, zero distance, empty mask.
    pub const NEUTRAL: Meta = Meta {
        phase: 0.0,
        distance: 0,
        event_mask: 0,
    };
}
