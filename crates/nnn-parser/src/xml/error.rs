//! XML-specific error kinds and source-position helpers.

/// Enumerates the failures that can occur while parsing or validating XML.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XmlErrorKind {
    /// The document contained bytes that were not valid UTF-8.
    InvalidUtf8,
    /// The parser reached end-of-input unexpectedly.
    Eof,
    /// A token appeared where the XML grammar did not allow it.
    UnexpectedToken,
    /// A tag or attribute name was malformed.
    InvalidTagName,
    /// An opening or closing tag was left unterminated.
    UnterminatedTag,
    /// A comment started but never reached a valid closing marker.
    UnterminatedComment,
    /// A CDATA section was opened but never closed.
    UnterminatedCdata,
    /// A processing instruction was opened but never closed.
    UnterminatedProcessingInstruction,
    /// A quoted attribute value was opened but never closed.
    UnterminatedAttributeValue,
    /// A closing tag did not match the currently open element.
    MismatchedClosingTag,
    /// The same attribute name appeared more than once on a single element.
    DuplicateAttribute,
    /// An XML entity or character reference was malformed.
    InvalidEntity,
    /// The XML declaration was malformed.
    InvalidDeclaration,
    /// The configured nesting depth limit was exceeded.
    MaxDepthExceeded,
    /// The configured maximum number of parsed nodes was exceeded.
    MaxNodeCountExceeded,
    /// An element exceeded the configured maximum number of attributes.
    MaxAttributeCountExceeded,
    /// An attribute value exceeded the configured maximum length.
    MaxAttributeValueLengthExceeded,
    /// An underlying filesystem or I/O operation failed.
    IoError,
}

/// Human-readable line and column information derived from a byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XmlErrorPosition {
    pub line: usize,
    pub column: usize,
}

/// Concrete XML error containing the error kind and offending byte offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XmlError {
    pub kind: XmlErrorKind,
    pub offset: usize,
}

impl XmlError {
    pub(crate) const fn new(kind: XmlErrorKind, offset: usize) -> Self {
        Self { kind, offset }
    }

    pub fn line_column(&self, input: &[u8]) -> XmlErrorPosition {
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
        XmlErrorPosition { line, column: col }
    }
}
