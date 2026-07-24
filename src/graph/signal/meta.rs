#[derive(Clone, Copy)]
pub(crate) struct Meta {
    pub phase: f32,
    pub distance: u32,
    pub event_mask: u32,
}

impl Meta {
    pub(crate) const NEUTRAL: Meta = Meta {
        phase: 0.0,
        distance: 0,
        event_mask: 0,
    };
}

const _: f32 = Meta::NEUTRAL.phase;
const _: u32 = Meta::NEUTRAL.event_mask;
