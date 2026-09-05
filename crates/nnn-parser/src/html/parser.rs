//! HTML parser implementation, depth checks, and strict validation helpers.

use alloc::{borrow::ToOwned, string::String, vec::Vec};

use super::entity::validate_entity;
use super::error::{HtmlError, HtmlErrorKind};
use super::lexer::Cursor;
use super::value::HtmlValue;

const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

const RAW_TEXT_ELEMENTS: &[&str] = &["script", "style"];

/// Default maximum nesting depth accepted by the HTML parser.
pub const DEFAULT_MAX_HTML_DEPTH: usize = 2048;

/// Resource limits used to bound HTML parsing on large or adversarial input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HtmlLimits {
    /// Maximum element nesting depth accepted during parsing.
    pub max_depth: usize,
    /// Maximum total number of nodes accepted in the document.
    pub max_node_count: usize,
    /// Maximum number of attributes allowed on a single element.
    pub max_attribute_count: usize,
    /// Maximum length allowed for a single attribute value.
    pub max_attribute_value_len: usize,
}

/// Conservative default limits for HTML parsing and validation.
pub const DEFAULT_HTML_LIMITS: HtmlLimits = HtmlLimits {
    max_depth: DEFAULT_MAX_HTML_DEPTH,
    max_node_count: 4 * 1024 * 1024,
    max_attribute_count: 4096,
    max_attribute_value_len: 8 * 1024 * 1024,
};

/// Stateful HTML parser configured through [`HtmlLimits`].
pub struct HtmlParser<'a> {
    cursor: Cursor<'a>,
    limits: HtmlLimits,
    nodes_seen: usize,
}

impl<'a> HtmlParser<'a> {
    /// Creates a parser with [`DEFAULT_HTML_LIMITS`].
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            limits: DEFAULT_HTML_LIMITS,
            nodes_seen: 0,
        }
    }

    /// Replaces the full parser limit set.
    pub const fn limits(mut self, limits: HtmlLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Overrides only the maximum nesting depth.
    pub const fn max_depth(mut self, max_depth: usize) -> Self {
        self.limits.max_depth = max_depth;
        self
    }

    /// Parses the document and returns a top-level [`HtmlValue`].
    pub fn parse(mut self) -> Result<HtmlValue<'a>, HtmlError> {
        self.parse_nodes(0)?;
        Ok(HtmlValue::Document)
    }

    /// Validates the document without retaining a DOM-like structure.
    pub fn validate(mut self) -> Result<(), HtmlError> {
        self.parse_nodes(0)?;
        Ok(())
    }

    fn tick_node(&mut self) -> Result<(), HtmlError> {
        self.nodes_seen = self.nodes_seen.saturating_add(1);
        if self.nodes_seen > self.limits.max_node_count {
            return Err(HtmlError::new(
                HtmlErrorKind::MaxNodeCountExceeded,
                self.cursor.position(),
            ));
        }
        Ok(())
    }

    fn parse_nodes(&mut self, depth: usize) -> Result<(), HtmlError> {
        if depth > self.limits.max_depth {
            return Err(HtmlError::new(
                HtmlErrorKind::MaxDepthExceeded,
                self.cursor.position(),
            ));
        }

        while !self.cursor.is_eof() {
            if self.cursor.peek() == Some(b'<') {
                if self.cursor.starts_with(b"</") {
                    return Ok(());
                }
                if self.cursor.starts_with(b"<!--") {
                    self.parse_comment()?;
                } else if self.cursor.starts_with(b"<!") {
                    self.parse_doctype()?;
                } else {
                    self.parse_element(depth)?;
                }
            } else {
                self.parse_text()?;
            }
        }

        Ok(())
    }

    fn parse_comment(&mut self) -> Result<(), HtmlError> {
        self.tick_node()?;
        let start = self.cursor.position();
        self.cursor.advance(4);

        loop {
            if self.cursor.is_eof() {
                return Err(HtmlError::new(HtmlErrorKind::UnterminatedComment, start));
            }
            if self.cursor.starts_with(b"-->") {
                self.cursor.advance(3);
                return Ok(());
            }
            self.cursor.advance(1);
        }
    }

    fn parse_doctype(&mut self) -> Result<(), HtmlError> {
        self.tick_node()?;
        let start = self.cursor.position();
        self.cursor.advance(2);

        loop {
            if self.cursor.is_eof() {
                return Err(HtmlError::new(HtmlErrorKind::UnterminatedDoctype, start));
            }
            if self.cursor.peek() == Some(b'>') {
                self.cursor.advance(1);
                return Ok(());
            }
            self.cursor.advance(1);
        }
    }

    fn parse_text(&mut self) -> Result<(), HtmlError> {
        self.tick_node()?;
        while let Some(b) = self.cursor.peek() {
            match b {
                b'<' => break,
                b'&' => validate_entity(&mut self.cursor)?,
                _ => self.cursor.advance(1),
            }
        }
        Ok(())
    }

    fn parse_element(&mut self, depth: usize) -> Result<(), HtmlError> {
        self.tick_node()?;
        let tag_start = self.cursor.position();
        self.cursor.advance(1);

        let tag_name = self.cursor.read_tag_name()?;

        self.parse_attributes()?;
        self.cursor.skip_ws();

        let self_closing = self.cursor.peek() == Some(b'/');
        if self_closing {
            self.cursor.advance(1);
        }

        if self.cursor.peek() != Some(b'>') {
            return Err(HtmlError::new(HtmlErrorKind::UnterminatedTag, tag_start));
        }
        self.cursor.advance(1);

        if self_closing || is_void_element(tag_name) {
            return Ok(());
        }

        if is_raw_text_element(tag_name) {
            return self.skip_raw_text(tag_name, tag_start);
        }

        self.parse_nodes(depth + 1)?;
        self.parse_closing_tag(tag_name, tag_start)
    }

    fn parse_attributes(&mut self) -> Result<(), HtmlError> {
        let mut count = 0usize;

        loop {
            self.cursor.skip_ws();
            match self.cursor.peek() {
                Some(b'>') | Some(b'/') | None => return Ok(()),
                _ => {}
            }

            self.parse_attribute()?;
            count = count.saturating_add(1);
            if count > self.limits.max_attribute_count {
                return Err(HtmlError::new(
                    HtmlErrorKind::MaxAttributeCountExceeded,
                    self.cursor.position(),
                ));
            }
        }
    }

    fn parse_attribute(&mut self) -> Result<(), HtmlError> {
        self.cursor.read_while(|b| {
            b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b':' || b == b'.'
        });

        self.cursor.skip_ws();

        if self.cursor.peek() != Some(b'=') {
            return Ok(());
        }
        self.cursor.advance(1);
        self.cursor.skip_ws();

        match self.cursor.peek() {
            Some(b'"') => self.parse_quoted_value(b'"'),
            Some(b'\'') => self.parse_quoted_value(b'\''),
            _ => {
                self.cursor.read_while(|b| {
                    !matches!(
                        b,
                        b' ' | b'\t' | b'\n' | b'\r' | b'>' | b'/' | b'"' | b'\'' | b'='
                    )
                });
                Ok(())
            }
        }
    }

    fn parse_quoted_value(&mut self, quote: u8) -> Result<(), HtmlError> {
        let start = self.cursor.position();
        self.cursor.advance(1);
        let content_start = self.cursor.position();

        loop {
            match self.cursor.peek() {
                None => return Err(HtmlError::new(HtmlErrorKind::UnterminatedAttribute, start)),
                Some(b) if b == quote => {
                    let len = self.cursor.position() - content_start;
                    if len > self.limits.max_attribute_value_len {
                        return Err(HtmlError::new(
                            HtmlErrorKind::MaxAttributeValueLengthExceeded,
                            content_start,
                        ));
                    }
                    self.cursor.advance(1);
                    return Ok(());
                }
                Some(b'&') => validate_entity(&mut self.cursor)?,
                _ => self.cursor.advance(1),
            }
        }
    }

    fn parse_closing_tag(&mut self, expected: &str, open_offset: usize) -> Result<(), HtmlError> {
        if !self.cursor.starts_with(b"</") {
            return Err(HtmlError::new(
                HtmlErrorKind::MismatchedClosingTag,
                self.cursor.position(),
            ));
        }
        self.cursor.advance(2);

        let close_name = self.cursor.read_tag_name()?;

        if !eq_ignore_ascii_case(close_name, expected) {
            return Err(HtmlError::new(
                HtmlErrorKind::MismatchedClosingTag,
                open_offset,
            ));
        }

        self.cursor.skip_ws();

        if self.cursor.peek() != Some(b'>') {
            return Err(HtmlError::new(
                HtmlErrorKind::UnterminatedTag,
                self.cursor.position(),
            ));
        }
        self.cursor.advance(1);
        Ok(())
    }

    fn skip_raw_text(&mut self, tag_name: &str, open_offset: usize) -> Result<(), HtmlError> {
        loop {
            if self.cursor.is_eof() {
                return Err(HtmlError::new(HtmlErrorKind::UnterminatedTag, open_offset));
            }
            if self.cursor.starts_with(b"</") {
                let saved = self.cursor.position();
                self.cursor.advance(2);
                if let Ok(name) = self.cursor.read_tag_name() {
                    if eq_ignore_ascii_case(name, tag_name) {
                        self.cursor.skip_ws();
                        if self.cursor.peek() == Some(b'>') {
                            self.cursor.advance(1);
                            return Ok(());
                        }
                    }
                }
                self.cursor.advance(0);
                if self.cursor.position() == saved + 2 {
                    self.cursor.advance(1);
                }
                continue;
            }
            self.cursor.advance(1);
        }
    }
}

fn is_void_element(name: &str) -> bool {
    VOID_ELEMENTS.iter().any(|&v| eq_ignore_ascii_case(v, name))
}

fn is_raw_text_element(name: &str) -> bool {
    RAW_TEXT_ELEMENTS
        .iter()
        .any(|&v| eq_ignore_ascii_case(v, name))
}

fn eq_ignore_ascii_case(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.bytes()
        .zip(b.bytes())
        .all(|(x, y)| x.eq_ignore_ascii_case(&y))
}

/// Parses HTML input using [`DEFAULT_HTML_LIMITS`].
pub fn parse_html(bytes: &[u8]) -> Result<HtmlValue<'_>, HtmlError> {
    HtmlParser::new(bytes).parse()
}

/// Performs fast HTML validation without keeping a parsed tree.
pub fn validate_html(bytes: &[u8]) -> Result<(), HtmlError> {
    HtmlParser::new(bytes).validate()
}

/// Validates a byte buffer without requiring any std filesystem API.
pub fn validate_html_file(bytes: &[u8]) -> Result<(), HtmlError> {
    let mut file_bytes = alloc::vec::Vec::new();
    crate::fs::read_file(bytes, &mut |chunk| {
        file_bytes.extend_from_slice(chunk);
        true
    })
    .map_err(|_| HtmlError::new(HtmlErrorKind::IoError, 0))?;
    validate_html(&file_bytes)
}

pub fn benchmark_validate_html_file(
    path: &[u8],
    output_dir: &[u8],
) -> Result<nnn_bmk::BenchmarkMetrics<'static>, HtmlError> {
    crate::fs::ensure_dir(output_dir)
        .map_err(|_| HtmlError::new(HtmlErrorKind::IoError, 0))?;
    let started_at = crate::fs::monotonic_ns();
    validate_html_file(path)?;
    Ok(crate::bmk::parser_metrics(
        "html",
        "utf-8",
        crate::fs::monotonic_ns().saturating_sub(started_at),
        0,
    ))
}

pub fn benchmark_exit(code: i32) -> ! {
    crate::fs::process_exit(code)
}

/// Extracts a lightweight list of tag and node markers from a HTML text input.
pub fn html_node_tags(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let doc = HtmlValue::Document;
    if doc.is_document() {
        out.push("document".to_owned());
    }
    let el = HtmlValue::Element;
    if el.is_element() {
        out.push("element".to_owned());
    }
    let comm = HtmlValue::Comment;
    if comm.is_comment() {
        out.push("comment".to_owned());
    }
    let dt = HtmlValue::Doctype;
    if dt.is_doctype() {
        out.push("doctype".to_owned());
    }
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let txt = HtmlValue::Text(trimmed);
            if let Some(s) = txt.as_text() {
                out.push(s.to_owned());
            }
        }
    }
    out
}

/// Performs stricter HTML validation, including duplicate-attribute and entity checks.
pub fn strict_validate_html(bytes: &[u8]) -> Result<(), HtmlError> {
    HtmlParser::new(bytes).validate()?;
    for (i, &b) in bytes.iter().enumerate() {
        if b == 0 {
            return Err(HtmlError::new(HtmlErrorKind::UnexpectedToken, i));
        }
    }
    if has_duplicate_html_attrs(bytes) {
        return Err(HtmlError::new(HtmlErrorKind::DuplicateAttribute, 0));
    }
    Ok(())
}

fn has_duplicate_html_attrs(bytes: &[u8]) -> bool {
    let text = match core::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(_) => return false,
    };
    for chunk in text.split('<').skip(1) {
        let tag = match chunk.split('>').next() {
            Some(t) => t,
            None => continue,
        };
        if tag.starts_with('/') || tag.starts_with('!') || tag.starts_with('?') {
            continue;
        }
        let mut names: Vec<String> = Vec::new();
        let b = tag.as_bytes();
        let mut i = 0;
        while i < b.len() && !b[i].is_ascii_whitespace() {
            i += 1;
        }
        while i < b.len() {
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= b.len() {
                break;
            }
            let start = i;
            while i < b.len() && b[i] != b'=' && !b[i].is_ascii_whitespace() && b[i] != b'/' {
                i += 1;
            }
            if i == start {
                i += 1;
                continue;
            }
            let name = tag[start..i].to_ascii_lowercase();
            if names.contains(&name) {
                return true;
            }
            names.push(name);
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < b.len() && b[i] == b'=' {
                i += 1;
                while i < b.len() && b[i].is_ascii_whitespace() {
                    i += 1;
                }
                if i < b.len() && (b[i] == b'"' || b[i] == b'\'') {
                    let q = b[i];
                    i += 1;
                    while i < b.len() && b[i] != q {
                        i += 1;
                    }
                    if i < b.len() {
                        i += 1;
                    }
                } else {
                    while i < b.len() && !b[i].is_ascii_whitespace() {
                        i += 1;
                    }
                }
            }
        }
    }
    false
}
