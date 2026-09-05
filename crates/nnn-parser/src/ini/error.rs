//! INI-specific error kinds and source-position helpers.

/// Enumerates the failures that can occur while parsing or validating INI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IniErrorKind {
    /// The input contained bytes that were not valid UTF-8.
    InvalidUtf8,
    /// A section header was opened but never closed properly.
    UnterminatedSection,
    /// A section header was present but its name was empty.
    EmptySectionName,
    /// A key was malformed or missing.
    InvalidKey,
    /// The same section appeared more than once when that was not allowed.
    DuplicateSection,
    /// The same key appeared more than once within a section.
    DuplicateKey,
    /// The configured maximum number of sections was exceeded.
    MaxSectionsExceeded,
    /// The configured maximum number of keys per section was exceeded.
    MaxKeysExceeded,
    /// A key exceeded the configured maximum length.
    MaxKeyLengthExceeded,
    /// A value exceeded the configured maximum length.
    MaxValueLengthExceeded,
    /// An underlying filesystem or I/O operation failed.
    IoError,
}

/// Human-readable line and column information derived from a byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IniErrorPosition {
    pub line: usize,
    pub column: usize,
}

/// Concrete INI error containing the error kind and offending byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IniError {
    pub kind: IniErrorKind,
    pub offset: usize,
}

impl IniError {
    pub(crate) const fn new(kind: IniErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    /// Converts the stored byte offset into a 1-based line and column pair.
    pub fn line_column(&self, input: &[u8]) -> IniErrorPosition {
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
        IniErrorPosition { line, column: col }
    }
}
