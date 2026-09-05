#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TxtErrorKind {
    IoError,
    InvalidUtf8,
    MaxTextLengthExceeded,
    MaxLinesExceeded,
    MaxLineLengthExceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxtErrorPosition {
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxtError {
    pub kind: TxtErrorKind,
    pub offset: usize,
}

impl TxtError {
    pub(crate) const fn new(kind: TxtErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    pub fn line_column(&self, input: &[u8]) -> TxtErrorPosition {
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

        TxtErrorPosition { line, column: col }
    }
}
