//! XML value markers returned by the format-specific parser.

/// Format-specific XML value markers returned by the parser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum XmlValue<'a> {
    /// Marker for a whole XML document.
    Document,
    /// Marker for an element node.
    Element,
    /// Borrowed text node content.
    Text(&'a str),
    /// Marker for a comment node.
    Comment,
    /// Borrowed CDATA payload.
    Cdata(&'a str),
    /// Marker for a processing instruction node.
    ProcessingInstruction,
}

impl<'a> XmlValue<'a> {
    /// Returns `true` when the value represents a whole document.
    pub const fn is_document(&self) -> bool {
        matches!(self, XmlValue::Document)
    }

    /// Returns `true` when the value marks an element node.
    pub const fn is_element(&self) -> bool {
        matches!(self, XmlValue::Element)
    }

    /// Returns `true` when the value marks a comment node.
    pub const fn is_comment(&self) -> bool {
        matches!(self, XmlValue::Comment)
    }

    /// Borrows the underlying text when the value is `Text`.
    pub const fn as_text(&self) -> Option<&str> {
        match self {
            XmlValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Borrows the underlying CDATA contents when the value is `Cdata`.
    pub const fn as_cdata(&self) -> Option<&str> {
        match self {
            XmlValue::Cdata(s) => Some(s),
            _ => None,
        }
    }
}
