#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CsvErrorKind {
    UnexpectedQuote,
    UnterminatedQuotedField,
    InvalidUtf8,
    TrailingCharactersAfterQuote,
    MaxRowsExceeded,
    MaxColumnsExceeded,
    MaxFieldLengthExceeded,
    MaxNodeCountExceeded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsvErrorPosition {
    pub line: usize,
    pub column: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsvError {
    pub kind: CsvErrorKind,
    pub offset: usize,
}

impl CsvError {
    pub(crate) const fn new(kind: CsvErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    pub fn line_column(&self, input: &[u8]) -> CsvErrorPosition {
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

        CsvErrorPosition { line, column: col }
    }
}
