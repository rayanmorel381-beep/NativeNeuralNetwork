pub struct Scratch<'a> {
    buf: &'a mut [u8],
    offset: usize,
}

impl<'a> Scratch<'a> {
    pub fn new(buf: &'a mut [u8]) -> Self {
        Scratch { buf, offset: 0 }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.offset]
    }

    pub fn base_ptr(&self) -> *const u8 {
        self.buf.as_ptr()
    }

    pub fn alloc_align(&mut self, len: usize, align: usize) -> Option<&mut [u8]> {
        if len == 0 {
            return Some(&mut []);
        }
        let align = align.max(1);
        let base_ptr = self.buf.as_ptr() as usize;
        let mut start = base_ptr.checked_add(self.offset)?;
        let rem = start % align;
        if rem != 0 {
            start = start.checked_add(align - rem)?;
        }
        let rel = start - base_ptr;
        if rel.saturating_add(len) > self.buf.len() {
            return None;
        }
        self.offset = rel.saturating_add(len);
        let s = &mut self.buf[rel..rel + len];
        Some(s)
    }
}
