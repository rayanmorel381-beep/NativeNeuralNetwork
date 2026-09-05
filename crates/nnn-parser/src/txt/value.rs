#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxtValue<'a> {
    Text(&'a str),
    Lines,
}

impl<'a> TxtValue<'a> {
    pub const fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(*v),
            Self::Lines => None,
        }
    }

    pub const fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    pub const fn is_lines(&self) -> bool {
        matches!(self, Self::Lines)
    }
}
