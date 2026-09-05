//! HTML value markers returned by the format-specific parser.

/// Format-specific HTML value markers returned by the parser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HtmlValue<'a> {
    /// Marker for a whole HTML document.
    Document,
    /// Marker for an element node.
    Element,
    /// Borrowed text node contents.
    Text(&'a str),
    /// Marker for a comment node.
    Comment,
    /// Marker for a doctype declaration.
    Doctype,
}

impl<'a> HtmlValue<'a> {
    /// Returns `true` when the value represents a whole document.
    pub const fn is_document(&self) -> bool {
        matches!(self, HtmlValue::Document)
    }

    /// Returns `true` when the value marks an element node.
    pub const fn is_element(&self) -> bool {
        matches!(self, HtmlValue::Element)
    }

    /// Returns `true` when the value marks a comment node.
    pub const fn is_comment(&self) -> bool {
        matches!(self, HtmlValue::Comment)
    }

    /// Returns `true` when the value marks a doctype node.
    pub const fn is_doctype(&self) -> bool {
        matches!(self, HtmlValue::Doctype)
    }

    /// Borrows the underlying text when the value is `Text`.
    pub const fn as_text(&self) -> Option<&str> {
        match self {
            HtmlValue::Text(v) => Some(v),
            _ => None,
        }
    }
}
