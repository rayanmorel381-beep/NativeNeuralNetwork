use super::error::{JsonError, JsonErrorKind};

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn len(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn bytes(&self) -> &'a [u8] {
        self.bytes
    }

    pub(crate) fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    pub(crate) fn advance(&mut self, n: usize) {
        self.pos = self.pos.saturating_add(n);
    }

    pub(crate) fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            match b {
                b' ' | b'\n' | b'\r' | b'\t' => self.pos += 1,
                _ => break,
            }
        }
    }

    pub(crate) fn consume(&mut self, expected: u8) -> Result<(), JsonError> {
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or(JsonError::new(JsonErrorKind::Eof, self.pos))?;
        if b != expected {
            return Err(JsonError::new(JsonErrorKind::UnexpectedToken, self.pos));
        }
        self.pos += 1;
        Ok(())
    }

    pub(crate) fn try_consume(&mut self, expected: u8) -> bool {
        if self.peek() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    pub(crate) fn expect_bytes(&mut self, expected: &[u8]) -> Result<(), JsonError> {
        if self.pos + expected.len() > self.bytes.len() {
            return Err(JsonError::new(JsonErrorKind::Eof, self.pos));
        }
        let got = &self.bytes[self.pos..self.pos + expected.len()];
        if got != expected {
            return Err(JsonError::new(JsonErrorKind::InvalidLiteral, self.pos));
        }
        self.pos += expected.len();
        Ok(())
    }
}
