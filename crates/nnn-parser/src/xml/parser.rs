//! XML parser implementation, declaration handling, and strict checks.

use alloc::{borrow::ToOwned, string::String, vec::Vec};

use super::error::{XmlError, XmlErrorKind};
use super::lexer::Cursor;
use super::value::XmlValue;

/// Resource limits used to bound XML parsing on large or untrusted input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XmlLimits {
    /// Maximum nesting depth allowed for element trees.
    pub max_depth: usize,
    /// Maximum total number of nodes allowed in the document.
    pub max_node_count: usize,
    /// Maximum number of attributes allowed on a single element.
    pub max_attribute_count: usize,
    /// Maximum length allowed for a single attribute value.
    pub max_attribute_value_len: usize,
}

/// Conservative default limits for XML parsing and validation.
pub const DEFAULT_XML_LIMITS: XmlLimits = XmlLimits {
    max_depth: 128,
    max_node_count: 256 * 1024,
    max_attribute_count: 256,
    max_attribute_value_len: 64 * 1024,
};

/// Stateful XML parser configured through [`XmlLimits`].
pub struct XmlParser<'a> {
    cursor: Cursor<'a>,
    limits: XmlLimits,
    nodes_seen: usize,
}

impl<'a> XmlParser<'a> {
    /// Creates a parser with [`DEFAULT_XML_LIMITS`].
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            limits: DEFAULT_XML_LIMITS,
            nodes_seen: 0,
        }
    }

    /// Replaces the full parser limit set.
    pub const fn limits(mut self, limits: XmlLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Overrides only the maximum element nesting depth.
    pub const fn max_depth(mut self, max_depth: usize) -> Self {
        self.limits.max_depth = max_depth;
        self
    }

    /// Parses the XML document and returns a top-level [`XmlValue`].
    pub fn parse(mut self) -> Result<XmlValue<'a>, XmlError> {
        self.cursor.skip_whitespace();
        self.skip_xml_declaration()?;
        self.cursor.skip_whitespace();
        self.parse_nodes(0)?;
        Ok(XmlValue::Document)
    }

    /// Validates XML structure without retaining a parsed node tree.
    pub fn validate(mut self) -> Result<(), XmlError> {
        self.cursor.skip_whitespace();
        self.skip_xml_declaration()?;
        self.cursor.skip_whitespace();
        self.parse_nodes(0)?;
        Ok(())
    }

    fn tick_node(&mut self) -> Result<(), XmlError> {
        self.nodes_seen = self.nodes_seen.saturating_add(1);
        if self.nodes_seen > self.limits.max_node_count {
            return Err(XmlError::new(
                XmlErrorKind::MaxNodeCountExceeded,
                self.cursor.position(),
            ));
        }
        Ok(())
    }

    fn skip_xml_declaration(&mut self) -> Result<(), XmlError> {
        if self.cursor.starts_with(b"<?xml") {
            self.skip_processing_instruction()?;
            self.cursor.skip_whitespace();
        }
        while self.cursor.starts_with(b"<!DOCTYPE") {
            self.skip_doctype()?;
            self.cursor.skip_whitespace();
        }
        Ok(())
    }

    fn skip_doctype(&mut self) -> Result<(), XmlError> {
        let start = self.cursor.position();
        self.cursor.advance(2);
        let mut depth = 1usize;
        loop {
            match self.cursor.peek() {
                None => return Err(XmlError::new(XmlErrorKind::UnterminatedTag, start)),
                Some(b'<') => {
                    depth += 1;
                    self.cursor.advance(1);
                }
                Some(b'>') => {
                    self.cursor.advance(1);
                    depth -= 1;
                    if depth == 0 {
                        return Ok(());
                    }
                }
                Some(_) => {
                    self.cursor.advance(1);
                }
            }
        }
    }

    fn parse_nodes(&mut self, depth: usize) -> Result<(), XmlError> {
        loop {
            self.cursor.skip_whitespace();
            if self.cursor.is_eof() {
                break;
            }
            if self.cursor.peek() != Some(b'<') {
                let text = self.cursor.read_text_until_lt()?;
                if !text.trim().is_empty() {
                    self.tick_node()?;
                }
                continue;
            }
            match self.cursor.peek_at(1) {
                Some(b'/') => break,
                Some(b'!') => {
                    if self.cursor.starts_with(b"<!--") {
                        self.parse_comment()?;
                    } else if self.cursor.starts_with(b"<![CDATA[") {
                        self.parse_cdata()?;
                    } else {
                        self.cursor.advance(2);
                        while !matches!(self.cursor.peek(), Some(b'>') | None) {
                            self.cursor.advance(1);
                        }
                        self.cursor.try_consume(b'>');
                    }
                }
                Some(b'?') => self.parse_processing_instruction()?,
                None => break,
                _ => {
                    if depth >= self.limits.max_depth {
                        return Err(XmlError::new(
                            XmlErrorKind::MaxDepthExceeded,
                            self.cursor.position(),
                        ));
                    }
                    self.parse_element(depth)?;
                }
            }
        }
        Ok(())
    }

    fn parse_element(&mut self, depth: usize) -> Result<(), XmlError> {
        self.tick_node()?;
        let start = self.cursor.position();
        self.cursor.advance(1);
        let tag_name = self.cursor.read_name()?;
        self.parse_attributes()?;
        self.cursor.skip_whitespace();

        if self.cursor.starts_with(b"/>") {
            self.cursor.advance(2);
            return Ok(());
        }

        self.cursor.expect(b'>')?;
        self.parse_nodes(depth + 1)?;

        if !self.cursor.starts_with(b"</") {
            return Err(XmlError::new(XmlErrorKind::UnterminatedTag, start));
        }
        self.cursor.advance(2);
        self.cursor.skip_whitespace();
        let closing = self.cursor.read_name()?;
        if closing != tag_name {
            return Err(XmlError::new(
                XmlErrorKind::MismatchedClosingTag,
                self.cursor.position(),
            ));
        }
        self.cursor.skip_whitespace();
        self.cursor.expect(b'>')?;
        Ok(())
    }

    fn parse_attributes(&mut self) -> Result<(), XmlError> {
        let mut count = 0usize;
        loop {
            self.cursor.skip_whitespace();
            match self.cursor.peek() {
                Some(b'>') | Some(b'/') | None => break,
                _ => {}
            }
            self.cursor.read_name()?;
            self.cursor.skip_whitespace();
            self.cursor.expect(b'=')?;
            self.cursor.skip_whitespace();
            let val = self.cursor.read_attr_value()?;
            if val.len() > self.limits.max_attribute_value_len {
                return Err(XmlError::new(
                    XmlErrorKind::MaxAttributeValueLengthExceeded,
                    self.cursor.position(),
                ));
            }
            count += 1;
            if count > self.limits.max_attribute_count {
                return Err(XmlError::new(
                    XmlErrorKind::MaxAttributeCountExceeded,
                    self.cursor.position(),
                ));
            }
        }
        Ok(())
    }

    fn parse_comment(&mut self) -> Result<(), XmlError> {
        let start = self.cursor.position();
        self.cursor.expect_str(b"<!--")?;
        self.tick_node()?;
        loop {
            if self.cursor.is_eof() {
                return Err(XmlError::new(XmlErrorKind::UnterminatedComment, start));
            }
            if self.cursor.starts_with(b"-->") {
                self.cursor.advance(3);
                return Ok(());
            }
            self.cursor.advance(1);
        }
    }

    fn parse_cdata(&mut self) -> Result<(), XmlError> {
        let start = self.cursor.position();
        self.cursor.expect_str(b"<![CDATA[")?;
        self.tick_node()?;
        let content_start = self.cursor.position();
        loop {
            if self.cursor.is_eof() {
                return Err(XmlError::new(XmlErrorKind::UnterminatedCdata, start));
            }
            if self.cursor.starts_with(b"]]>") {
                let content_len = self.cursor.position().saturating_sub(content_start);
                if content_len > self.limits.max_attribute_value_len {
                    return Err(XmlError::new(
                        XmlErrorKind::MaxAttributeValueLengthExceeded,
                        start,
                    ));
                }
                self.cursor.advance(3);
                return Ok(());
            }
            self.cursor.advance(1);
        }
    }

    fn parse_processing_instruction(&mut self) -> Result<(), XmlError> {
        self.skip_processing_instruction()
    }

    fn skip_processing_instruction(&mut self) -> Result<(), XmlError> {
        let start = self.cursor.position();
        self.cursor.advance(2);
        loop {
            if self.cursor.is_eof() {
                return Err(XmlError::new(
                    XmlErrorKind::UnterminatedProcessingInstruction,
                    start,
                ));
            }
            if self.cursor.starts_with(b"?>") {
                self.cursor.advance(2);
                return Ok(());
            }
            self.cursor.advance(1);
        }
    }
}

/// Parses XML input using [`DEFAULT_XML_LIMITS`].
pub fn parse_xml(bytes: &[u8]) -> Result<XmlValue<'_>, XmlError> {
    XmlParser::new(bytes).parse()
}

/// Performs fast XML validation without keeping the parsed node tree.
pub fn validate_xml(bytes: &[u8]) -> Result<(), XmlError> {
    XmlParser::new(bytes).validate()
}

/// Validates XML input from a byte buffer without using std filesystem APIs.
pub fn validate_xml_file(bytes: &[u8]) -> Result<(), XmlError> {
    let mut file_bytes = alloc::vec::Vec::new();
    crate::fs::read_file(bytes, &mut |chunk| {
        file_bytes.extend_from_slice(chunk);
        true
    })
    .map_err(|_| XmlError::new(XmlErrorKind::IoError, 0))?;
    validate_xml(&file_bytes)
}

pub fn benchmark_validate_xml_file(
    path: &[u8],
    output_dir: &[u8],
) -> Result<nnn_bmk::BenchmarkMetrics<'static>, XmlError> {
    crate::fs::ensure_dir(output_dir)
        .map_err(|_| XmlError::new(XmlErrorKind::IoError, 0))?;
    let started_at = crate::fs::monotonic_ns();
    validate_xml_file(path)?;
    Ok(crate::bmk::parser_metrics(
        "xml",
        "utf-8",
        crate::fs::monotonic_ns().saturating_sub(started_at),
        0,
    ))
}

pub fn benchmark_exit(code: i32) -> ! {
    crate::fs::process_exit(code)
}

/// Extracts lightweight tag and node markers from an XML text input.
pub fn xml_node_tags(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let doc = XmlValue::Document;
    if doc.is_document() {
        out.push("document".to_owned());
    }
    let el = XmlValue::Element;
    if el.is_element() {
        out.push("element".to_owned());
    }
    let comm = XmlValue::Comment;
    if comm.is_comment() {
        out.push("comment".to_owned());
    }
    let pi = XmlValue::ProcessingInstruction;
    if !pi.is_element() {
        out.push("processing_instruction".to_owned());
    }
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let txt = XmlValue::Text(trimmed);
            if let Some(s) = txt.as_text() {
                out.push(s.to_owned());
            }
            let cd = XmlValue::Cdata(trimmed);
            if let Some(s) = cd.as_cdata() {
                out.push(s.to_owned());
            }
        }
    }
    out
}

/// Performs stricter XML validation, including declaration, entity, and duplicate-attribute checks.
pub fn strict_validate_xml(bytes: &[u8]) -> Result<(), XmlError> {
    XmlParser::new(bytes).validate()?;
    let text =
        core::str::from_utf8(bytes).map_err(|_| XmlError::new(XmlErrorKind::InvalidUtf8, 0))?;
    check_xml_decl(text)?;
    check_xml_comments(text)?;
    check_xml_entities(text)?;
    check_single_xml_root(text)?;
    if has_duplicate_xml_attrs(text) {
        return Err(XmlError::new(XmlErrorKind::DuplicateAttribute, 0));
    }
    Ok(())
}

fn check_xml_comments(text: &str) -> Result<(), XmlError> {
    let bytes = text.as_bytes();
    let mut idx = 0usize;

    while idx < bytes.len() {
        if bytes[idx..].starts_with(b"<!--") {
            let start = idx;
            idx += 4;
            let body_start = idx;
            while idx + 3 <= bytes.len() && &bytes[idx..idx + 3] != b"-->" {
                idx += 1;
            }
            if idx + 3 > bytes.len() {
                return Err(XmlError::new(XmlErrorKind::UnterminatedComment, start));
            }
            if bytes[body_start..idx]
                .windows(2)
                .any(|window| window == b"--")
            {
                return Err(XmlError::new(XmlErrorKind::UnterminatedComment, start));
            }
            idx += 3;
            continue;
        }
        idx += 1;
    }

    Ok(())
}

fn check_xml_decl(text: &str) -> Result<(), XmlError> {
    if let Some(rest) = text.strip_prefix("<?xml") {
        let end = rest
            .find("?>")
            .ok_or(XmlError::new(XmlErrorKind::InvalidDeclaration, 0))?;
        let decl = &rest[..end];
        if !decl.contains("version") {
            return Err(XmlError::new(XmlErrorKind::InvalidDeclaration, 0));
        }
    }
    Ok(())
}

fn check_xml_entities(text: &str) -> Result<(), XmlError> {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut in_tag = false;
    let mut in_cdata = false;
    while i < bytes.len() {
        if in_cdata {
            if bytes[i..].starts_with(b"]]>") {
                in_cdata = false;
                i += 3;
            } else {
                i += 1;
            }
            continue;
        }
        if bytes[i..].starts_with(b"<![CDATA[") {
            in_cdata = true;
            i += 9;
            continue;
        }
        if bytes[i] == b'<' {
            in_tag = true;
            i += 1;
            continue;
        }
        if bytes[i] == b'>' {
            in_tag = false;
            i += 1;
            continue;
        }
        if !in_tag && bytes[i] == b'&' {
            let start = i;
            i += 1;
            while i < bytes.len()
                && bytes[i] != b';'
                && bytes[i] != b'<'
                && !bytes[i].is_ascii_whitespace()
            {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b';' {
                return Err(XmlError::new(XmlErrorKind::InvalidEntity, start));
            }
            i += 1;
        } else {
            i += 1;
        }
    }
    Ok(())
}

fn check_single_xml_root(text: &str) -> Result<(), XmlError> {
    let bytes = text.as_bytes();
    let mut idx = 0usize;
    let mut depth = 0usize;
    let mut root_count = 0usize;

    while idx < bytes.len() {
        if bytes[idx] != b'<' {
            if depth == 0 && !bytes[idx].is_ascii_whitespace() {
                return Err(XmlError::new(XmlErrorKind::UnexpectedToken, idx));
            }
            idx += 1;
            continue;
        }

        if bytes[idx..].starts_with(b"<!--") {
            idx = find_xml_marker_end(bytes, idx + 4, b"-->")?;
            continue;
        }
        if bytes[idx..].starts_with(b"<![CDATA[") {
            idx = find_xml_marker_end(bytes, idx + 9, b"]]>")?;
            continue;
        }
        if bytes[idx..].starts_with(b"<?") {
            idx = find_xml_marker_end(bytes, idx + 2, b"?>")?;
            continue;
        }
        if bytes[idx..].starts_with(b"<!DOCTYPE") {
            idx = find_doctype_end(bytes, idx + 9)?;
            continue;
        }
        if bytes[idx..].starts_with(b"</") {
            depth = depth.saturating_sub(1);
            idx = find_tag_end(bytes, idx + 2)?;
            continue;
        }

        if depth == 0 {
            root_count += 1;
            if root_count > 1 {
                return Err(XmlError::new(XmlErrorKind::UnexpectedToken, idx));
            }
        }

        let self_closing = is_self_closing_tag(bytes, idx + 1)?;
        if !self_closing {
            depth += 1;
        }
        idx = find_tag_end(bytes, idx + 1)?;
    }

    if root_count != 1 || depth != 0 {
        return Err(XmlError::new(XmlErrorKind::UnexpectedToken, bytes.len()));
    }

    Ok(())
}

fn find_xml_marker_end(bytes: &[u8], start: usize, marker: &[u8]) -> Result<usize, XmlError> {
    let mut idx = start;
    while idx + marker.len() <= bytes.len() {
        if &bytes[idx..idx + marker.len()] == marker {
            return Ok(idx + marker.len());
        }
        idx += 1;
    }
    Err(XmlError::new(XmlErrorKind::UnexpectedToken, start))
}

fn find_doctype_end(bytes: &[u8], start: usize) -> Result<usize, XmlError> {
    let mut idx = start;
    let mut in_quote: Option<u8> = None;
    let mut bracket_depth = 0usize;

    while idx < bytes.len() {
        let b = bytes[idx];
        if let Some(quote) = in_quote {
            if b == quote {
                in_quote = None;
            }
            idx += 1;
            continue;
        }

        match b {
            b'\'' | b'"' => in_quote = Some(b),
            b'[' => bracket_depth = bracket_depth.saturating_add(1),
            b']' if bracket_depth > 0 => bracket_depth -= 1,
            b'>' if bracket_depth == 0 => return Ok(idx + 1),
            _ => {}
        }
        idx += 1;
    }

    Err(XmlError::new(XmlErrorKind::UnexpectedToken, start))
}

fn find_tag_end(bytes: &[u8], start: usize) -> Result<usize, XmlError> {
    let mut idx = start;
    let mut in_quote: Option<u8> = None;

    while idx < bytes.len() {
        let b = bytes[idx];
        if let Some(quote) = in_quote {
            if b == quote {
                in_quote = None;
            }
            idx += 1;
            continue;
        }

        match b {
            b'\'' | b'"' => in_quote = Some(b),
            b'>' => return Ok(idx + 1),
            _ => {}
        }
        idx += 1;
    }

    Err(XmlError::new(XmlErrorKind::UnexpectedToken, start))
}

fn is_self_closing_tag(bytes: &[u8], start: usize) -> Result<bool, XmlError> {
    let end = find_tag_end(bytes, start)?;
    let mut idx = end.saturating_sub(1);
    while idx > start && bytes[idx - 1].is_ascii_whitespace() {
        idx -= 1;
    }
    Ok(idx > start && bytes[idx - 1] == b'/')
}

fn has_duplicate_xml_attrs(text: &str) -> bool {
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
        while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'/' {
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
            let name = tag[start..i].to_owned();
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
