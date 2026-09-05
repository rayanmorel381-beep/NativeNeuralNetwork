//! Cursor primitives and token-scanning helpers for TOML.

use super::error::{TomlError, TomlErrorKind};

/// Byte cursor used by the TOML parser for token-level scanning.
pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    /// Creates a new cursor positioned at the start of the TOML document.
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Returns the current byte offset.
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Returns `true` when the cursor has reached end-of-input.
    pub const fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    /// Peeks the current byte without consuming it.
    pub fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    /// Advances by up to `n` bytes, saturating at end-of-input.
    pub fn advance(&mut self, n: usize) {
        self.pos = core::cmp::min(self.pos.saturating_add(n), self.bytes.len());
    }

    /// Consumes and returns the next byte.
    pub fn next_byte(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    /// Skips spaces and tabs on the current line.
    pub fn skip_whitespace_inline(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t')) {
            self.pos += 1;
        }
    }

    /// Skips any ASCII whitespace, including line breaks.
    pub fn skip_whitespace_and_newlines(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    /// Skips an optional trailing comment and the following newline sequence.
    pub fn skip_comment_and_newline(&mut self) {
        self.skip_whitespace_inline();
        if self.peek() == Some(b'#') {
            while !matches!(self.peek(), Some(b'\n') | None) {
                self.pos += 1;
            }
        }
        if self.peek() == Some(b'\r') {
            self.pos += 1;
        }
        if self.peek() == Some(b'\n') {
            self.pos += 1;
        }
    }

    /// Reads an unquoted TOML key.
    pub fn read_bare_key(&mut self) -> Result<&'a str, TomlError> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'-' || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(TomlError::new(TomlErrorKind::InvalidKey, self.pos));
        }
        core::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| TomlError::new(TomlErrorKind::InvalidUtf8, start))
    }

    /// Reads a double-quoted TOML basic string and validates escapes superficially.
    pub fn read_basic_string(&mut self) -> Result<&'a str, TomlError> {
        let start_pos = self.pos;
        self.pos += 1;
        let content_start = self.pos;
        loop {
            match self.peek() {
                None => return Err(TomlError::new(TomlErrorKind::UnterminatedString, start_pos)),
                Some(b'"') => {
                    let end = self.pos;
                    self.pos += 1;
                    let raw = core::str::from_utf8(&self.bytes[content_start..end])
                        .map_err(|_| TomlError::new(TomlErrorKind::InvalidUtf8, content_start))?;
                    return Ok(raw);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"' | b'\\' | b'n' | b't' | b'r' | b'b' | b'f') => {
                            self.pos += 1;
                        }
                        Some(b'u') => {
                            self.pos += 5;
                        }
                        Some(b'U') => {
                            self.pos += 9;
                        }
                        _ => return Err(TomlError::new(TomlErrorKind::InvalidEscape, self.pos)),
                    }
                }
                Some(_) => {
                    self.pos += 1;
                }
            }
        }
    }

    /// Reads a single-quoted TOML literal string.
    pub fn read_literal_string(&mut self) -> Result<&'a str, TomlError> {
        let start_pos = self.pos;
        self.pos += 1;
        let content_start = self.pos;
        loop {
            match self.peek() {
                None => return Err(TomlError::new(TomlErrorKind::UnterminatedString, start_pos)),
                Some(b'\'') => {
                    let end = self.pos;
                    self.pos += 1;
                    let raw = core::str::from_utf8(&self.bytes[content_start..end])
                        .map_err(|_| TomlError::new(TomlErrorKind::InvalidUtf8, content_start))?;
                    return Ok(raw);
                }
                Some(_) => {
                    self.pos += 1;
                }
            }
        }
    }

    /// Reads either a basic or literal string based on the opening quote.
    pub fn read_string(&mut self) -> Result<&'a str, TomlError> {
        match self.peek() {
            Some(b'"') => self.read_basic_string(),
            Some(b'\'') => self.read_literal_string(),
            _ => Err(TomlError::new(TomlErrorKind::InvalidString, self.pos)),
        }
    }

    /// Reads a TOML key in either quoted or bare form.
    pub fn read_key(&mut self) -> Result<&'a str, TomlError> {
        match self.peek() {
            Some(b'"') => self.read_basic_string(),
            Some(b'\'') => self.read_literal_string(),
            _ => self.read_bare_key(),
        }
    }

    /// Reads the raw token used later to distinguish numbers from date/time values.
    pub fn read_number_or_date_raw(&mut self) -> &'a str {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric()
                || matches!(b, b'+' | b'-' | b'.' | b'_' | b':' | b'T' | b'Z')
            {
                self.pos += 1;
            } else {
                break;
            }
        }
        core::str::from_utf8(&self.bytes[start..self.pos]).unwrap_or("")
    }

    /// Consumes `expected` or returns a structured TOML error.
    pub fn expect(&mut self, expected: u8) -> Result<(), TomlError> {
        match self.peek() {
            Some(b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(_) => Err(TomlError::new(TomlErrorKind::UnexpectedToken, self.pos)),
            None => Err(TomlError::new(TomlErrorKind::Eof, self.pos)),
        }
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
