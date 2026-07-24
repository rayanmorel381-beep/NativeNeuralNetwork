use super::error::{JsonError, JsonErrorKind};
use super::lexer::Cursor;
use super::number::parse_number;
use super::string::parse_string;
use super::value::JsonValue;
use crate::engine::runtime::FixedSliceVec;

const DUP_TRACK_LIMIT: usize = 256;

pub const DEFAULT_MAX_DEPTH: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DuplicateKeyPolicy {
    Allow,
    Reject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonLimits {
    pub max_depth: usize,
    pub max_string_len: usize,
    pub max_array_len: usize,
    pub max_object_len: usize,
    pub max_node_count: usize,
    pub duplicate_key_policy: DuplicateKeyPolicy,
}

pub const DEFAULT_LIMITS: JsonLimits = JsonLimits {
    max_depth: DEFAULT_MAX_DEPTH,
    max_string_len: 64 * 1024,
    max_array_len: 16 * 1024,
    max_object_len: 16 * 1024,
    max_node_count: 128 * 1024,
    duplicate_key_policy: DuplicateKeyPolicy::Allow,
};

pub struct JsonParser<'a> {
    cursor: Cursor<'a>,
    limits: JsonLimits,
    nodes_seen: usize,
}

impl<'a> JsonParser<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            limits: DEFAULT_LIMITS,
            nodes_seen: 0,
        }
    }

    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.limits.max_depth = max_depth;
        self
    }

    pub const fn with_limits(mut self, limits: JsonLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn parse(mut self) -> Result<JsonValue<'a>, JsonError> {
        self.cursor.skip_ws();
        let value = self.parse_value(0)?;
        self.cursor.skip_ws();
        if self.cursor.position() != self.cursor.len() {
            return Err(JsonError::new(
                JsonErrorKind::TrailingCharacters,
                self.cursor.position(),
            ));
        }
        Ok(value)
    }

    pub fn validate(mut self) -> Result<(), JsonError> {
        self.cursor.skip_ws();
        self.parse_value(0)?;
        self.cursor.skip_ws();
        if self.cursor.position() != self.cursor.len() {
            return Err(JsonError::new(
                JsonErrorKind::TrailingCharacters,
                self.cursor.position(),
            ));
        }
        Ok(())
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue<'a>, JsonError> {
        if depth > self.limits.max_depth {
            return Err(JsonError::new(
                JsonErrorKind::MaxDepthExceeded,
                self.cursor.position(),
            ));
        }

        self.nodes_seen = self.nodes_seen.saturating_add(1);
        if self.nodes_seen > self.limits.max_node_count {
            return Err(JsonError::new(
                JsonErrorKind::MaxNodeCountExceeded,
                self.cursor.position(),
            ));
        }

        let b = self
            .cursor
            .peek()
            .ok_or(JsonError::new(JsonErrorKind::Eof, self.cursor.position()))?;

        match b {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => {
                let s = parse_string(&mut self.cursor)?;
                if s.raw.len() > self.limits.max_string_len {
                    return Err(JsonError::new(
                        JsonErrorKind::MaxStringLengthExceeded,
                        self.cursor.position(),
                    ));
                }
                Ok(JsonValue::String(s.raw))
            }
            b't' => self.parse_true(),
            b'f' => self.parse_false(),
            b'n' => self.parse_null(),
            b'-' | b'0'..=b'9' => parse_number(&mut self.cursor).map(JsonValue::Number),
            _ => Err(JsonError::new(
                JsonErrorKind::UnexpectedToken,
                self.cursor.position(),
            )),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue<'a>, JsonError> {
        self.cursor.consume(b'{')?;
        self.cursor.skip_ws();

        if self.cursor.try_consume(b'}') {
            return Ok(JsonValue::Object);
        }

        let mut object_len = 0usize;
        let mut seen_key_ranges_buf = [(0usize, 0usize); DUP_TRACK_LIMIT];
        let mut seen_key_ranges = FixedSliceVec::new(&mut seen_key_ranges_buf);

        loop {
            self.cursor.skip_ws();
            let key = parse_string(&mut self.cursor)?;
            if key.raw.len() > self.limits.max_string_len {
                return Err(JsonError::new(
                    JsonErrorKind::MaxStringLengthExceeded,
                    self.cursor.position(),
                ));
            }

            if self.limits.duplicate_key_policy == DuplicateKeyPolicy::Reject {
                let key_bytes = &self.cursor.bytes()[key.start..key.end];
                if !seen_key_ranges.is_empty() {
                    let ranges = seen_key_ranges.as_slice();
                    let mut i = 0usize;
                    while i < seen_key_ranges.len() {
                        let (start, end) = ranges[i];
                        if &self.cursor.bytes()[start..end] == key_bytes {
                            return Err(JsonError::new(
                                JsonErrorKind::DuplicateObjectKey,
                                self.cursor.position(),
                            ));
                        }
                        i += 1;
                    }
                }

                if seen_key_ranges.is_full() {
                    return Err(JsonError::new(
                        JsonErrorKind::MaxObjectLengthExceeded,
                        self.cursor.position(),
                    ));
                }

                if seen_key_ranges.push((key.start, key.end)).is_err() {
                    return Err(JsonError::new(
                        JsonErrorKind::MaxObjectLengthExceeded,
                        self.cursor.position(),
                    ));
                }
            }

            self.cursor.skip_ws();
            self.cursor.consume(b':')?;
            self.cursor.skip_ws();
            self.parse_value(depth)?;
            object_len = object_len.saturating_add(1);
            if object_len > self.limits.max_object_len {
                return Err(JsonError::new(
                    JsonErrorKind::MaxObjectLengthExceeded,
                    self.cursor.position(),
                ));
            }
            self.cursor.skip_ws();

            if self.cursor.try_consume(b',') {
                self.cursor.skip_ws();
                continue;
            }
            self.cursor.consume(b'}')?;
            return Ok(JsonValue::Object);
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue<'a>, JsonError> {
        self.cursor.consume(b'[')?;
        self.cursor.skip_ws();

        if self.cursor.try_consume(b']') {
            return Ok(JsonValue::Array);
        }

        let mut array_len = 0usize;

        loop {
            self.cursor.skip_ws();
            self.parse_value(depth)?;
            array_len = array_len.saturating_add(1);
            if array_len > self.limits.max_array_len {
                return Err(JsonError::new(
                    JsonErrorKind::MaxArrayLengthExceeded,
                    self.cursor.position(),
                ));
            }
            self.cursor.skip_ws();

            if self.cursor.try_consume(b',') {
                self.cursor.skip_ws();
                continue;
            }
            self.cursor.consume(b']')?;
            return Ok(JsonValue::Array);
        }
    }

    fn parse_true(&mut self) -> Result<JsonValue<'a>, JsonError> {
        self.cursor.expect_bytes(b"true")?;
        Ok(JsonValue::Bool(true))
    }

    fn parse_false(&mut self) -> Result<JsonValue<'a>, JsonError> {
        self.cursor.expect_bytes(b"false")?;
        Ok(JsonValue::Bool(false))
    }

    fn parse_null(&mut self) -> Result<JsonValue<'a>, JsonError> {
        self.cursor.expect_bytes(b"null")?;
        Ok(JsonValue::Null)
    }
}

pub fn parse_json(bytes: &[u8]) -> Result<JsonValue<'_>, JsonError> {
    JsonParser::new(bytes).parse()
}

pub fn parse_json_with_max_depth(
    bytes: &[u8],
    max_depth: usize,
) -> Result<JsonValue<'_>, JsonError> {
    JsonParser::new(bytes).with_max_depth(max_depth).parse()
}

pub fn parse_json_with_limits(
    bytes: &[u8],
    limits: JsonLimits,
) -> Result<JsonValue<'_>, JsonError> {
    JsonParser::new(bytes).with_limits(limits).parse()
}

pub fn validate_json(bytes: &[u8]) -> Result<(), JsonError> {
    JsonParser::new(bytes).validate()
}
