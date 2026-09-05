//! TOML parser implementation, nesting checks, and strict validation rules.

use alloc::{borrow::ToOwned, string::String, vec::Vec};

use super::error::{TomlError, TomlErrorKind};
use super::lexer::Cursor;
use super::value::TomlValue;

/// Resource limits used to bound TOML parsing on large or adversarial input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TomlLimits {
    /// Maximum allowed nesting depth across tables, arrays, and inline tables.
    pub max_depth: usize,
    /// Maximum allowed length for a parsed key.
    pub max_key_len: usize,
    /// Maximum allowed length for a parsed string literal.
    pub max_string_len: usize,
    /// Maximum number of elements accepted in a single array.
    pub max_array_len: usize,
    /// Maximum number of entries accepted in a single table.
    pub max_table_len: usize,
    /// Maximum total node count accepted during the parse.
    pub max_node_count: usize,
}

/// Conservative default limits for TOML parsing and validation.
pub const DEFAULT_TOML_LIMITS: TomlLimits = TomlLimits {
    max_depth: 64,
    max_key_len: 256,
    max_string_len: 64 * 1024,
    max_array_len: 16 * 1024,
    max_table_len: 16 * 1024,
    max_node_count: 128 * 1024,
};

/// Stateful TOML parser configured through [`TomlLimits`].
pub struct TomlParser<'a> {
    cursor: Cursor<'a>,
    limits: TomlLimits,
    nodes_seen: usize,
}

impl<'a> TomlParser<'a> {
    /// Creates a parser with [`DEFAULT_TOML_LIMITS`].
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            limits: DEFAULT_TOML_LIMITS,
            nodes_seen: 0,
        }
    }

    /// Replaces the full parser limit set.
    pub const fn limits(mut self, limits: TomlLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Parses the input into a format-specific [`TomlValue`] tree.
    pub fn parse(mut self) -> Result<TomlValue<'a>, TomlError> {
        self.parse_document()?;
        Ok(TomlValue::Table)
    }

    /// Validates TOML structure without retaining the parsed tree.
    pub fn validate(mut self) -> Result<(), TomlError> {
        self.parse_document()
    }

    fn tick_node(&mut self) -> Result<(), TomlError> {
        self.nodes_seen = self.nodes_seen.saturating_add(1);
        if self.nodes_seen > self.limits.max_node_count {
            return Err(TomlError::new(
                TomlErrorKind::MaxNodeCountExceeded,
                self.cursor.position(),
            ));
        }
        Ok(())
    }

    fn parse_document(&mut self) -> Result<(), TomlError> {
        loop {
            self.cursor.skip_whitespace_and_newlines();
            self.cursor.skip_comment_and_newline();
            if self.cursor.is_eof() {
                break;
            }
            match self.cursor.peek() {
                Some(b'[') => self.parse_table_header()?,
                Some(b'#') => {
                    self.cursor.skip_comment_and_newline();
                }
                Some(b'\n' | b'\r') => {
                    self.cursor.advance(1);
                }
                None => break,
                _ => self.parse_keyval(0)?,
            }
        }
        Ok(())
    }

    fn parse_table_header(&mut self) -> Result<(), TomlError> {
        self.tick_node()?;
        self.cursor.advance(1);
        let array_table = self.cursor.try_consume(b'[');
        self.cursor.skip_whitespace_inline();
        self.parse_key()?;
        self.cursor.skip_whitespace_inline();
        self.cursor.expect(b']')?;
        if array_table {
            self.cursor.expect(b']')?;
        }
        self.cursor.skip_comment_and_newline();
        Ok(())
    }

    fn parse_keyval(&mut self, depth: usize) -> Result<(), TomlError> {
        if depth > self.limits.max_depth {
            return Err(TomlError::new(
                TomlErrorKind::MaxDepthExceeded,
                self.cursor.position(),
            ));
        }
        self.cursor.skip_whitespace_inline();
        self.parse_key()?;
        self.cursor.skip_whitespace_inline();
        self.cursor.expect(b'=')?;
        self.cursor.skip_whitespace_inline();
        self.parse_value(depth)?;
        self.cursor.skip_comment_and_newline();
        Ok(())
    }

    fn parse_key(&mut self) -> Result<(), TomlError> {
        let key = self.cursor.read_key()?;
        if key.len() > self.limits.max_key_len {
            return Err(TomlError::new(
                TomlErrorKind::MaxKeyLengthExceeded,
                self.cursor.position(),
            ));
        }

        let mut segments = 1usize;
        self.cursor.skip_whitespace_inline();
        while self.cursor.try_consume(b'.') {
            segments = segments.saturating_add(1);
            if segments.saturating_sub(1) > self.limits.max_depth {
                return Err(TomlError::new(
                    TomlErrorKind::MaxDepthExceeded,
                    self.cursor.position(),
                ));
            }
            self.cursor.skip_whitespace_inline();
            let dotted = self.cursor.read_key()?;
            if dotted.len() > self.limits.max_key_len {
                return Err(TomlError::new(
                    TomlErrorKind::MaxKeyLengthExceeded,
                    self.cursor.position(),
                ));
            }
            self.cursor.skip_whitespace_inline();
        }
        Ok(())
    }

    fn parse_value(&mut self, depth: usize) -> Result<TomlValue<'a>, TomlError> {
        if depth > self.limits.max_depth {
            return Err(TomlError::new(
                TomlErrorKind::MaxDepthExceeded,
                self.cursor.position(),
            ));
        }
        self.tick_node()?;
        match self.cursor.peek() {
            Some(b'"' | b'\'') => {
                let s = self.cursor.read_string()?;
                if s.len() > self.limits.max_string_len {
                    return Err(TomlError::new(
                        TomlErrorKind::MaxStringLengthExceeded,
                        self.cursor.position(),
                    ));
                }
                Ok(TomlValue::String(s))
            }
            Some(b't') => {
                self.expect_keyword(b"true")?;
                Ok(TomlValue::Bool(true))
            }
            Some(b'f') => {
                self.expect_keyword(b"false")?;
                Ok(TomlValue::Bool(false))
            }
            Some(b'[') => self.parse_array(depth + 1),
            Some(b'{') => self.parse_inline_table(depth + 1),
            Some(b) if b.is_ascii_digit() || b == b'-' || b == b'+' => self.parse_number_or_date(),
            _ => Err(TomlError::new(
                TomlErrorKind::UnexpectedToken,
                self.cursor.position(),
            )),
        }
    }

    fn expect_keyword(&mut self, kw: &[u8]) -> Result<(), TomlError> {
        let start = self.cursor.position();
        let len = kw.len();
        for &b in kw {
            match self.cursor.peek() {
                Some(got) if got == b => {
                    self.cursor.advance(1);
                }
                Some(_) => return Err(TomlError::new(TomlErrorKind::UnexpectedToken, start)),
                None => return Err(TomlError::new(TomlErrorKind::Eof, start + len)),
            }
        }
        Ok(())
    }

    fn parse_number_or_date(&mut self) -> Result<TomlValue<'a>, TomlError> {
        let pos = self.cursor.position();
        let raw = self.cursor.read_number_or_date_raw();
        if raw.is_empty() {
            return Err(TomlError::new(TomlErrorKind::InvalidNumber, pos));
        }
        let clean: String = raw.chars().filter(|&c| c != '_').collect();
        if let Ok(i) = clean.parse::<i64>() {
            return Ok(TomlValue::Integer(i));
        }
        if let Ok(f) = clean.parse::<f64>() {
            return Ok(TomlValue::Float(f));
        }
        Ok(TomlValue::String(raw))
    }

    fn parse_array(&mut self, depth: usize) -> Result<TomlValue<'a>, TomlError> {
        self.cursor.advance(1);
        let mut count = 0usize;
        loop {
            self.cursor.skip_whitespace_and_newlines();
            self.skip_comments_and_whitespace();
            if self.cursor.try_consume(b']') {
                return Ok(TomlValue::Array);
            }
            self.parse_value(depth)?;
            count = count.saturating_add(1);
            if count > self.limits.max_array_len {
                return Err(TomlError::new(
                    TomlErrorKind::MaxArrayLengthExceeded,
                    self.cursor.position(),
                ));
            }
            self.cursor.skip_whitespace_and_newlines();
            self.skip_comments_and_whitespace();
            if !self.cursor.try_consume(b',') {
                self.cursor.skip_whitespace_and_newlines();
                self.skip_comments_and_whitespace();
                if self.cursor.try_consume(b']') {
                    return Ok(TomlValue::Array);
                }
                return Err(TomlError::new(
                    TomlErrorKind::UnterminatedArray,
                    self.cursor.position(),
                ));
            }
        }
    }

    fn parse_inline_table(&mut self, depth: usize) -> Result<TomlValue<'a>, TomlError> {
        self.cursor.advance(1);
        self.cursor.skip_whitespace_inline();
        let mut count = 0usize;
        if self.cursor.try_consume(b'}') {
            return Ok(TomlValue::InlineTable);
        }
        loop {
            self.parse_key()?;
            self.cursor.skip_whitespace_inline();
            self.cursor.expect(b'=')?;
            self.cursor.skip_whitespace_inline();
            self.parse_value(depth)?;
            count = count.saturating_add(1);
            if count > self.limits.max_table_len {
                return Err(TomlError::new(
                    TomlErrorKind::MaxTableLengthExceeded,
                    self.cursor.position(),
                ));
            }
            self.cursor.skip_whitespace_inline();
            if self.cursor.try_consume(b'}') {
                return Ok(TomlValue::InlineTable);
            }
            self.cursor.expect(b',')?;
            self.cursor.skip_whitespace_inline();
        }
    }

    fn skip_comments_and_whitespace(&mut self) {
        loop {
            self.cursor.skip_whitespace_and_newlines();
            if self.cursor.peek() == Some(b'#') {
                while !matches!(self.cursor.peek(), Some(b'\n') | None) {
                    self.cursor.next_byte();
                }
            } else {
                break;
            }
        }
    }
}

/// Parses TOML input using [`DEFAULT_TOML_LIMITS`].
pub fn parse_toml(bytes: &[u8]) -> Result<TomlValue<'_>, TomlError> {
    TomlParser::new(bytes).parse()
}

/// Performs fast TOML validation without keeping the parsed tree.
pub fn validate_toml(bytes: &[u8]) -> Result<(), TomlError> {
    TomlParser::new(bytes).validate()
}

/// Validates TOML input from a byte buffer without using std filesystem APIs.
pub fn validate_toml_file(bytes: &[u8]) -> Result<(), TomlError> {
    let mut file_bytes = alloc::vec::Vec::new();
    crate::fs::read_file(bytes, &mut |chunk| {
        file_bytes.extend_from_slice(chunk);
        true
    })
    .map_err(|_| TomlError::new(TomlErrorKind::IoError, 0))?;
    validate_toml(&file_bytes)
}

pub fn benchmark_validate_toml_file(
    path: &[u8],
    output_dir: &[u8],
) -> Result<nnn_bmk::BenchmarkMetrics<'static>, TomlError> {
    crate::fs::ensure_dir(output_dir)
        .map_err(|_| TomlError::new(TomlErrorKind::IoError, 0))?;
    let started_at = crate::fs::monotonic_ns();
    validate_toml_file(path)?;
    Ok(crate::bmk::parser_metrics(
        "toml",
        "utf-8",
        crate::fs::monotonic_ns().saturating_sub(started_at),
        0,
    ))
}

pub fn benchmark_exit(code: i32) -> ! {
    crate::fs::process_exit(code)
}

/// Performs stricter TOML validation, including duplicate-key and date checks.
pub fn strict_validate_toml(bytes: &[u8]) -> Result<(), TomlError> {
    TomlParser::new(bytes).validate()?;
    let text =
        core::str::from_utf8(bytes).map_err(|_| TomlError::new(TomlErrorKind::InvalidUtf8, 0))?;
    let mut tables: Vec<String> = Vec::new();
    let mut current_keys: Vec<String> = Vec::new();
    for (offset, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("[[") && trimmed.ends_with("]]") {
            current_keys.clear();
        } else if trimmed.starts_with('[') && trimmed.ends_with(']') && !trimmed.starts_with("[[") {
            let name = trimmed[1..trimmed.len() - 1].trim().to_owned();
            if tables.contains(&name) {
                return Err(TomlError::new(TomlErrorKind::DuplicateTable, offset));
            }
            tables.push(name);
            current_keys.clear();
        } else if let Some(eq) = trimmed.find('=') {
            let key = trimmed[..eq].trim().to_owned();
            if current_keys.contains(&key) {
                return Err(TomlError::new(TomlErrorKind::DuplicateKey, offset));
            }
            current_keys.push(key);
            let val = trimmed[eq + 1..].trim();
            if val.starts_with('{') && !val.ends_with('}') {
                return Err(TomlError::new(
                    TomlErrorKind::UnterminatedInlineTable,
                    offset,
                ));
            }
            if looks_like_toml_date(val) && !is_valid_toml_date(val) {
                return Err(TomlError::new(TomlErrorKind::InvalidDate, offset));
            }
        }
    }
    Ok(())
}

fn looks_like_toml_date(s: &str) -> bool {
    s.len() >= 10
        && s.as_bytes()[4] == b'-'
        && s.as_bytes()[7] == b'-'
        && s[..4].bytes().all(|b| b.is_ascii_digit())
}

fn is_valid_toml_date(s: &str) -> bool {
    let date_part = s.split(['T', 't', ' ']).next().unwrap_or("");
    let parts: Vec<&str> = date_part.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    let month = parts[1].parse::<u32>().unwrap_or(0);
    let day = parts[2].parse::<u32>().unwrap_or(0);
    (1..=12).contains(&month) && (1..=31).contains(&day)
}
