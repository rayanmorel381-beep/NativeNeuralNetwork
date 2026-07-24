#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JsonErrorKind {
    Eof,
    TrailingCharacters,
    UnexpectedToken,
    InvalidLiteral,
    InvalidNumber,
    InvalidEscape,
    InvalidUnicodeEscape,
    InvalidUnicodeSurrogate,
    ControlCharacterInString,
    InvalidUtf8,
    MaxDepthExceeded,
    MaxStringLengthExceeded,
    MaxArrayLengthExceeded,
    MaxObjectLengthExceeded,
    MaxNodeCountExceeded,
    DuplicateObjectKey,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonErrorPosition {
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonError {
    pub kind: JsonErrorKind,
    pub offset: usize,
}

impl JsonError {
    pub(crate) const fn new(kind: JsonErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    pub fn line_column(&self, input: &[u8]) -> JsonErrorPosition {
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

        JsonErrorPosition { line, column: col }
    }
}
