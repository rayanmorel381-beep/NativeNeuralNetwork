#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum YamlErrorKind {
    Eof,
    InvalidIndentation,
    UnexpectedToken,
    InvalidMappingKey,
    InvalidScalar,
    UnsupportedFeature,
    MaxDepthExceeded,
    MaxScalarLengthExceeded,
    MaxSequenceLengthExceeded,
    MaxMappingLengthExceeded,
    MaxNodeCountExceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YamlErrorPosition {
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YamlError {
    pub kind: YamlErrorKind,
    pub offset: usize,
}

impl YamlError {
    pub(crate) const fn new(kind: YamlErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    pub fn line_column(&self, input: &[u8]) -> YamlErrorPosition {
        let end = core::cmp::min(self.offset, input.len());
        let mut line = 1usize;
        let mut col = 1usize;
        let mut idx = 0usize;

        while idx < end {
            if input[idx] == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
            idx += 1;
        }

        YamlErrorPosition { line, column: col }
    }
}
