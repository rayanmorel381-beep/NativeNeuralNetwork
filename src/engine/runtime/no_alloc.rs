pub struct FixedSliceVec<'a, T> {
    buf: &'a mut [T],
    len: usize,
}

impl<'a, T> FixedSliceVec<'a, T> {
    pub fn new(buf: &'a mut [T]) -> Self {
        Self { buf, len: 0 }
    }

    pub fn push(&mut self, value: T) -> Result<(), T> {
        if self.len >= self.buf.len() {
            return Err(value);
        }
        self.buf[self.len] = value;
        self.len += 1;
        Ok(())
    }

    pub fn as_slice(&self) -> &[T] {
        &self.buf[..self.len]
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_full(&self) -> bool {
        self.len >= self.buf.len()
    }
}

pub struct FixedWriter<'a> {
    buf: &'a mut [u8],
    len: usize,
    overflow: bool,
}

impl<'a> FixedWriter<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Self {
            buf,
            len: 0,
            overflow: false,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn overflowed(&self) -> bool {
        self.overflow
    }
}

impl<'a> core::fmt::Write for FixedWriter<'a> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let end = match self.len.checked_add(bytes.len()) {
            Some(v) if v <= self.buf.len() => v,
            _ => {
                self.overflow = true;
                return Err(core::fmt::Error);
            }
        };
        self.buf[self.len..end].copy_from_slice(bytes);
        self.len = end;
        Ok(())
    }
}
