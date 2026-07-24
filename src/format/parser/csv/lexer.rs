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

    pub const fn bytes(&self) -> &'a [u8] {
        self.bytes
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
}
