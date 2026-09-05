//! HTML-specific error kinds and line/column reporting.

/// Enumerates the failures that can occur while parsing or validating HTML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HtmlErrorKind {
    /// The parser reached end-of-input unexpectedly.
    Eof,
    /// A token appeared where the HTML grammar did not allow it.
    UnexpectedToken,
    /// A tag name was malformed.
    InvalidTagName,
    /// A tag was opened but never properly closed.
    UnterminatedTag,
    /// A comment started but never reached a valid closing marker.
    UnterminatedComment,
    /// An attribute declaration was malformed or unfinished.
    UnterminatedAttribute,
    /// A doctype declaration was opened but never closed.
    UnterminatedDoctype,
    /// An HTML entity or character reference was malformed.
    UnterminatedEntity,
    /// The same attribute appeared more than once on a single element.
    DuplicateAttribute,
    /// A closing tag did not match the currently open element.
    MismatchedClosingTag,
    /// The input contained bytes that were not valid UTF-8.
    InvalidUtf8,
    /// The configured maximum nesting depth was exceeded.
    MaxDepthExceeded,
    /// The configured maximum number of parsed nodes was exceeded.
    MaxNodeCountExceeded,
    /// An element exceeded the configured maximum number of attributes.
    MaxAttributeCountExceeded,
    /// An attribute value exceeded the configured maximum allowed length.
    MaxAttributeValueLengthExceeded,
    /// An underlying filesystem or I/O operation failed.
    IoError,
}

/// Human-readable line and column information derived from a byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HtmlErrorPosition {
    pub line: usize,
    pub column: usize,
}

/// Concrete HTML error containing the error kind and offending byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HtmlError {
    pub kind: HtmlErrorKind,
    pub offset: usize,
}

impl HtmlError {
    pub(crate) const fn new(kind: HtmlErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    /// Converts the stored byte offset into a 1-based line and column pair.
    pub fn line_column(&self, input: &[u8]) -> HtmlErrorPosition {
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

        HtmlErrorPosition { line, column: col }
    }
}
