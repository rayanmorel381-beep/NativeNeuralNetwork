//! Line-oriented scanning helpers for INI documents.

use super::error::{IniError, IniErrorKind};

/// Significant INI line returned by the line cursor.
pub struct IniLine<'a> {
    pub content: &'a str,
    pub offset: usize,
}

/// Sequential line cursor that skips blank and comment-only INI lines.
pub struct LineCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> LineCursor<'a> {
    /// Creates a new line cursor over the raw document bytes.
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    /// Returns the current byte offset in the source document.
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Returns the next meaningful INI line, skipping blanks and comments.
    pub fn next_line(&mut self) -> Result<Option<IniLine<'a>>, IniError> {
        loop {
            if self.pos >= self.bytes.len() {
                return Ok(None);
            }
            let start = self.pos;
            let end = find_line_end(self.bytes, start);
            self.pos = if end < self.bytes.len() { end + 1 } else { end };

            let line_bytes = &self.bytes[start..end];
            let trimmed_bytes = trim_ascii_end(line_bytes);

            let content = core::str::from_utf8(trimmed_bytes)
                .map_err(|_| IniError::new(IniErrorKind::InvalidUtf8, start))?;
            let content = content.trim_start();

            if content.is_empty() || content.starts_with(';') || content.starts_with('#') {
                continue;
            }

            return Ok(Some(IniLine {
                content,
                offset: start,
            }));
        }
    }
}

fn find_line_end(bytes: &[u8], mut start: usize) -> usize {
    while start < bytes.len() {
        if bytes[start] == b'\n' {
            return start;
        }
        start += 1;
    }
    start
}

fn trim_ascii_end(bytes: &[u8]) -> &[u8] {
    let mut end = bytes.len();
    while end > 0 && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\r' || bytes[end - 1] == b'\t')
    {
        end -= 1;
    }
    &bytes[..end]
}
