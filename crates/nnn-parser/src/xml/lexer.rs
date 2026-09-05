//! Low-level cursor primitives for scanning XML input.

use super::error::{XmlError, XmlErrorKind};

/// Byte cursor used by the XML parser for token-level scanning.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Creates a new cursor positioned at the start of the XML input.
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Returns the current byte offset.
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Returns `true` when no more bytes are available.
    pub const fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Peeks the current byte without consuming it.
    pub fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Peeks a byte at `offset` from the current position.
    pub fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    /// Advances by up to `n` bytes, saturating at end-of-input.
    pub fn advance(&mut self, n: usize) {
        self.pos = core::cmp::min(self.pos.saturating_add(n), self.bytes.len());
    }

    /// Consumes XML-significant ASCII whitespace between tokens.
    pub fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Returns `true` when the remaining input starts with `needle`.
    pub fn starts_with(&self, needle: &[u8]) -> bool {
        self.bytes
            .get(self.pos..)
            .is_some_and(|s| s.starts_with(needle))
    }

    /// Reads an XML name according to the simplified parser rules.
    pub fn read_name(&mut self) -> Result<&'a str, XmlError> {
        let start = self.pos;
        match self.peek() {
            Some(b) if is_name_start(b) => {
                self.pos += 1;
            }
            _ => return Err(XmlError::new(XmlErrorKind::InvalidTagName, self.pos)),
        }
        while let Some(b) = self.peek() {
            if is_name_char(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        core::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| XmlError::new(XmlErrorKind::InvalidUtf8, start))
    }

    /// Reads a quoted XML attribute value.
    pub fn read_attr_value(&mut self) -> Result<&'a str, XmlError> {
        let start_pos = self.pos;
        let quote = self
            .peek()
            .ok_or(XmlError::new(XmlErrorKind::Eof, self.pos))?;
        if quote != b'"' && quote != b'\'' {
            return Err(XmlError::new(
                XmlErrorKind::UnterminatedAttributeValue,
                self.pos,
            ));
        }
        self.pos += 1;
        let content_start = self.pos;
        loop {
            match self.peek() {
                None => {
                    return Err(XmlError::new(
                        XmlErrorKind::UnterminatedAttributeValue,
                        start_pos,
                    ));
                }
                Some(b) if b == quote => {
                    let end = self.pos;
                    self.pos += 1;
                    let raw = core::str::from_utf8(&self.bytes[content_start..end])
                        .map_err(|_| XmlError::new(XmlErrorKind::InvalidUtf8, content_start))?;
                    return Ok(raw);
                }
                Some(_) => {
                    self.pos += 1;
                }
            }
        }
    }

    /// Reads character data until the next `<` delimiter.
    pub fn read_text_until_lt(&mut self) -> Result<&'a str, XmlError> {
        let start = self.pos;
        while !matches!(self.peek(), Some(b'<') | None) {
            self.pos += 1;
        }
        core::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| XmlError::new(XmlErrorKind::InvalidUtf8, start))
    }

    /// Consumes `expected` or returns a structured XML error.
    pub fn expect(&mut self, expected: u8) -> Result<(), XmlError> {
        match self.peek() {
            Some(b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(_) => Err(XmlError::new(XmlErrorKind::UnexpectedToken, self.pos)),
            None => Err(XmlError::new(XmlErrorKind::Eof, self.pos)),
        }
    }

    /// Consumes a fixed byte string from the current position.
    pub fn expect_str(&mut self, s: &[u8]) -> Result<(), XmlError> {
        for &b in s {
            self.expect(b)?;
        }
        Ok(())
    }

    /// Consumes `expected` if present and reports success as a boolean.
    pub fn try_consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }
}

pub(crate) fn is_name_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b':'
}

pub(crate) fn is_name_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b':')
}
