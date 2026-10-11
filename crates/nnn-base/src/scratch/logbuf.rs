//! Diagnostics that fit in the memory already carved.
//!
//! A [`LogBuf`] writes bytes into a fixed buffer and never allocates. It exists
//! because a memory pillar that debugs itself must not need memory it does not
//! have: the buffer is taken from the arena like any other carve, through
//! [`LogBuf::from_carve`], so diagnostics cost exactly what the caller decided
//! to spend.
//!
//! # Truncation is counted, not hidden
//!
//! When a write does not fit, the bytes that did not fit are added to
//! [`dropped`](LogBuf::dropped) and [`write_bytes`](LogBuf::write_bytes) returns
//! how many were actually written. Nothing is discarded silently and nothing
//! panics: a full buffer is a reportable condition, so a caller can assert on it
//! in a test and ignore it in production. [`clear`](LogBuf::clear) resets both
//! the length and the counter.
//!
//! # Format integration
//!
//! The [`fmt::Write`] implementation lets this buffer be used directly with
//! `write!` and `format_args`. A truncated write returns [`fmt::Error`], which
//! matches what the standard library's own writers do, so ordinary formatting
//! code behaves as expected.

use core::fmt;

use crate::scratch::carve::{Carve, CarveError};

pub struct LogBuf<'a> {
    buffer: &'a mut [u8],
    len: usize,
    dropped: usize,
}

impl<'a> LogBuf<'a> {
    pub fn from_carve(carve: Carve<'a>) -> Result<Self, CarveError> {
        if carve.is_empty() {
            return Err(CarveError::SizeMismatch);
        }

        Ok(Self::new(carve.data))
    }

    pub fn new(
        buffer: &'a mut [u8],
    ) -> Self {
        Self {
            buffer,
            len: 0,
            dropped: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn dropped(&self) -> usize {
        self.dropped
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.dropped = 0;
    }

    pub fn write_bytes(
        &mut self,
        bytes: &[u8],
    ) -> usize {
        let available = self.buffer.len().saturating_sub(self.len);
        let written = bytes.len().min(available);

        if written < bytes.len() {
            self.dropped = self
                .dropped
                .saturating_add(bytes.len() - written);
        }

        if written != 0 {
            let start = self.len;
            let end = start + written;

            self.buffer[start..end]
                .copy_from_slice(&bytes[..written]);

            self.len = end;
        }

        written
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer[..self.len]
    }

    pub fn write_fmt(
        &mut self,
        args: fmt::Arguments<'_>,
    ) -> fmt::Result {
        fmt::Write::write_fmt(self, args)
    }
}

impl<'a> fmt::Write for LogBuf<'a> {
    fn write_str(
        &mut self,
        value: &str,
    ) -> fmt::Result {
        let written = self.write_bytes(value.as_bytes());

        if written < value.len() {
            return Err(fmt::Error);
        }

        Ok(())
    }
}