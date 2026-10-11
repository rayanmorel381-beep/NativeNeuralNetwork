/// How an edge transports the signal that carries it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Dense,
    Sequential,
    Wave,
    Event,
}

impl Mode {
    /// Every mode, in encoding order.
    pub const ALL: [Mode; 4] = [Mode::Dense, Mode::Sequential, Mode::Wave, Mode::Event];

    /// The mode's stable byte encoding.
    pub fn as_u8(self) -> u8 {
        match self {
            Mode::Dense => 0,
            Mode::Sequential => 1,
            Mode::Wave => 2,
            Mode::Event => 3,
        }
    }

    /// The mode for `value`, or `None` when it is not a known encoding.
    pub fn from_u8(value: u8) -> Option<Mode> {
        match value {
            0 => Some(Mode::Dense),
            1 => Some(Mode::Sequential),
            2 => Some(Mode::Wave),
            3 => Some(Mode::Event),
            _ => None,
        }
    }
}
