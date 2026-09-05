//! TOML-specific error kinds and source-position helpers.

/// Enumerates the failures that can occur while parsing or validating TOML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TomlErrorKind {
    /// The document contained bytes that were not valid UTF-8.
    InvalidUtf8,
    /// The parser reached end-of-input unexpectedly.
    Eof,
    /// A token appeared where the TOML grammar did not allow it.
    UnexpectedToken,
    /// A bare or quoted key was malformed.
    InvalidKey,
    /// A string literal was malformed.
    InvalidString,
    /// A string escape sequence was invalid.
    InvalidEscape,
    /// A numeric literal was malformed.
    InvalidNumber,
    /// A date or date-time literal was malformed.
    InvalidDate,
    /// A string literal was opened but never closed.
    UnterminatedString,
    /// An array literal was opened but never closed.
    UnterminatedArray,
    /// An inline table was opened but never closed.
    UnterminatedInlineTable,
    /// The same key was defined more than once in the same scope.
    DuplicateKey,
    /// The same table was declared more than once incompatibly.
    DuplicateTable,
    /// The configured nesting depth limit was exceeded.
    MaxDepthExceeded,
    /// A parsed key exceeded the configured maximum length.
    MaxKeyLengthExceeded,
    /// A parsed string exceeded the configured maximum length.
    MaxStringLengthExceeded,
    /// An array exceeded the configured maximum number of items.
    MaxArrayLengthExceeded,
    /// A table exceeded the configured maximum number of entries.
    MaxTableLengthExceeded,
    /// The overall node-count limit for the parse was exceeded.
    MaxNodeCountExceeded,
    /// An underlying filesystem or I/O operation failed.
    IoError,
}

/// Human-readable line and column information derived from a byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TomlErrorPosition {
    pub line: usize,
    pub column: usize,
}

/// Concrete TOML error containing the error kind and offending byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TomlError {
    pub kind: TomlErrorKind,
    pub offset: usize,
}

impl TomlError {
    pub(crate) const fn new(kind: TomlErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    pub fn line_column(&self, input: &[u8]) -> TomlErrorPosition {
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
        TomlErrorPosition { line, column: col }
    }
}
