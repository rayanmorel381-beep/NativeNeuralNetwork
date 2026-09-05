use super::error::{YamlError, YamlErrorKind};

#[derive(Clone, Copy)]
pub struct YamlLine<'a> {
    pub indent: usize,
    pub content: &'a str,
    pub offset: usize,
}

pub struct LineCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> LineCursor<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    pub const fn position(&self) -> usize {
        self.pos
    }

    pub fn peek(&self) -> Result<Option<YamlLine<'a>>, YamlError> {
        self.scan_from(self.pos)
    }

    pub fn next(&mut self) -> Result<Option<YamlLine<'a>>, YamlError> {
        let line = self.scan_from(self.pos)?;
        if line.is_some() {
            self.advance_to_next_significant_line()?;
        }
        Ok(line)
    }

    fn advance_to_next_significant_line(&mut self) -> Result<(), YamlError> {
        while self.pos < self.bytes.len() {
            let line_end = find_line_end(self.bytes, self.pos);
            self.pos = if line_end < self.bytes.len() {
                line_end + 1
            } else {
                line_end
            };

            if self.scan_from(self.pos)?.is_some() {
                return Ok(());
            }
        }
        Ok(())
    }

    fn scan_from(&self, mut start: usize) -> Result<Option<YamlLine<'a>>, YamlError> {
        while start < self.bytes.len() {
            let line_end = find_line_end(self.bytes, start);
            let line_bytes = &self.bytes[start..line_end];

            let mut idx = 0usize;
            let mut indent = 0usize;
            while idx < line_bytes.len() {
                match line_bytes[idx] {
                    b' ' => {
                        indent += 1;
                        idx += 1;
                    }
                    b'\t' => {
                        return Err(YamlError::new(
                            YamlErrorKind::InvalidIndentation,
                            start + idx,
                        ));
                    }
                    _ => break,
                }
            }

            let content_start = idx;
            if content_start >= line_bytes.len() {
                start = if line_end < self.bytes.len() {
                    line_end + 1
                } else {
                    line_end
                };
                continue;
            }

            if line_bytes[content_start] == b'#' {
                start = if line_end < self.bytes.len() {
                    line_end + 1
                } else {
                    line_end
                };
                continue;
            }

            let mut content_end = line_bytes.len();
            while content_end > content_start && line_bytes[content_end - 1] == b' ' {
                content_end -= 1;
            }

            let content_bytes = &line_bytes[content_start..content_end];
            let content = core::str::from_utf8(content_bytes).map_err(|_| {
                YamlError::new(YamlErrorKind::UnexpectedToken, start + content_start)
            })?;

            return Ok(Some(YamlLine {
                indent,
                content,
                offset: start + content_start,
            }));
        }

        Ok(None)
    }
}

fn find_line_end(bytes: &[u8], mut start: usize) -> usize {
    while start < bytes.len() {
        if bytes[start] == b'\n' {
            return start;
        }
        start += 1;
    }
    bytes.len()
}
