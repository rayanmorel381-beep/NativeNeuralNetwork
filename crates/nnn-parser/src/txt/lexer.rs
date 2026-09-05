use super::error::{TxtError, TxtErrorKind};

pub struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub const fn position(&self) -> usize {
        self.pos
    }

    pub const fn len(&self) -> usize {
        self.bytes.len()
    }

    pub const fn is_eof(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    pub fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    pub fn advance(&mut self, n: usize) {
        self.pos = core::cmp::min(self.pos.saturating_add(n), self.bytes.len());
    }

    pub fn next(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    pub fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    pub fn read_line(&mut self) -> Option<&'a str> {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b == b'\n' {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start && self.peek().is_none() {
            return None;
        }
        let end = self.pos;
        if self.peek() == Some(b'\n') {
            self.pos += 1;
        }
        let slice = &self.bytes[start..end];
        core::str::from_utf8(slice).ok()
    }

    pub fn read_while<F>(&mut self, mut predicate: F) -> Option<&'a str>
    where
        F: FnMut(u8) -> bool,
    {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if predicate(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return None;
        }
        let slice = &self.bytes[start..self.pos];
        core::str::from_utf8(slice).ok()
    }

    pub fn expect_byte(&mut self, expected: u8) -> Result<(), TxtError> {
        match self.peek() {
            Some(b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            Some(_) => Err(TxtError::new(TxtErrorKind::InvalidUtf8, self.pos)),
            None => Err(TxtError::new(TxtErrorKind::InvalidUtf8, self.pos)),
        }
    }
}
