#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Dense,
    Sequential,
    Wave,
    Event,
}

impl Mode {
    pub(crate) const ALL: [Mode; 4] = [Mode::Dense, Mode::Sequential, Mode::Wave, Mode::Event];

    pub(crate) fn as_u8(self) -> u8 {
        match self {
            Mode::Dense => 0,
            Mode::Sequential => 1,
            Mode::Wave => 2,
            Mode::Event => 3,
        }
    }

    pub(crate) fn from_u8(value: u8) -> Option<Mode> {
        match value {
            0 => Some(Mode::Dense),
            1 => Some(Mode::Sequential),
            2 => Some(Mode::Wave),
            3 => Some(Mode::Event),
            _ => None,
        }
    }
}

const _: usize = Mode::ALL.len();
