use super::error::{YamlError, YamlErrorKind};
use super::lexer::{LineCursor, YamlLine};
use super::scalar::parse_scalar;
use super::value::YamlValue;

pub const DEFAULT_MAX_YAML_DEPTH: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YamlLimits {
    pub max_depth: usize,
    pub max_scalar_len: usize,
    pub max_sequence_len: usize,
    pub max_mapping_len: usize,
    pub max_node_count: usize,
}

pub const DEFAULT_YAML_LIMITS: YamlLimits = YamlLimits {
    max_depth: DEFAULT_MAX_YAML_DEPTH,
    max_scalar_len: 64 * 1024,
    max_sequence_len: 16 * 1024,
    max_mapping_len: 16 * 1024,
    max_node_count: 128 * 1024,
};

pub struct YamlParser<'a> {
    cursor: LineCursor<'a>,
    limits: YamlLimits,
    nodes_seen: usize,
}

impl<'a> YamlParser<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: LineCursor::new(bytes),
            limits: DEFAULT_YAML_LIMITS,
            nodes_seen: 0,
        }
    }

    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.limits.max_depth = max_depth;
        self
    }

    pub const fn with_limits(mut self, limits: YamlLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn parse(mut self) -> Result<YamlValue<'a>, YamlError> {
        let first = self
            .cursor
            .peek()?
            .ok_or(YamlError::new(YamlErrorKind::Eof, self.cursor.position()))?;
        self.parse_node(first.indent, 0)
    }

    pub fn validate(mut self) -> Result<(), YamlError> {
        let first = self
            .cursor
            .peek()?
            .ok_or(YamlError::new(YamlErrorKind::Eof, self.cursor.position()))?;
        self.parse_node(first.indent, 0)?;
        if self.cursor.peek()?.is_some() {
            let line = self.cursor.peek()?.expect("peek checked is_some");
            return Err(YamlError::new(YamlErrorKind::UnexpectedToken, line.offset));
        }
        Ok(())
    }

    fn parse_node(&mut self, base_indent: usize, depth: usize) -> Result<YamlValue<'a>, YamlError> {
        if depth > self.limits.max_depth {
            return Err(YamlError::new(
                YamlErrorKind::MaxDepthExceeded,
                self.cursor.position(),
            ));
        }

        self.nodes_seen = self.nodes_seen.saturating_add(1);
        if self.nodes_seen > self.limits.max_node_count {
            return Err(YamlError::new(
                YamlErrorKind::MaxNodeCountExceeded,
                self.cursor.position(),
            ));
        }

        let line = self
            .cursor
            .peek()?
            .ok_or(YamlError::new(YamlErrorKind::Eof, self.cursor.position()))?;

        if line.indent < base_indent {
            return Err(YamlError::new(
                YamlErrorKind::InvalidIndentation,
                line.offset,
            ));
        }
        if line.indent > base_indent {
            return Err(YamlError::new(
                YamlErrorKind::InvalidIndentation,
                line.offset,
            ));
        }

        if is_sequence_entry(line.content) {
            self.parse_sequence(base_indent, depth + 1)
        } else if has_mapping_separator(line.content) {
            self.parse_mapping(base_indent, depth + 1)
        } else {
            self.parse_scalar_line(line)
        }
    }

    fn parse_scalar_line(&mut self, line: YamlLine<'a>) -> Result<YamlValue<'a>, YamlError> {
        if line.content.len() > self.limits.max_scalar_len {
            return Err(YamlError::new(
                YamlErrorKind::MaxScalarLengthExceeded,
                line.offset,
            ));
        }
        self.cursor.next()?;
        parse_scalar(line.content, line.offset)
    }

    fn parse_sequence(
        &mut self,
        base_indent: usize,
        depth: usize,
    ) -> Result<YamlValue<'a>, YamlError> {
        let mut count = 0usize;

        loop {
            let Some(line) = self.cursor.peek()? else {
                break;
            };
            if line.indent < base_indent {
                break;
            }
            if line.indent > base_indent {
                return Err(YamlError::new(
                    YamlErrorKind::InvalidIndentation,
                    line.offset,
                ));
            }
            if !is_sequence_entry(line.content) {
                break;
            }

            let item_text = line.content[1..].trim_start();
            self.cursor.next()?;

            if !item_text.is_empty() {
                if item_text.len() > self.limits.max_scalar_len {
                    return Err(YamlError::new(
                        YamlErrorKind::MaxScalarLengthExceeded,
                        line.offset,
                    ));
                }
                parse_scalar(item_text, line.offset)?;
            } else {
                let nested = self
                    .cursor
                    .peek()?
                    .ok_or(YamlError::new(YamlErrorKind::Eof, line.offset))?;
                if nested.indent <= base_indent {
                    return Err(YamlError::new(
                        YamlErrorKind::InvalidIndentation,
                        nested.offset,
                    ));
                }
                self.parse_node(nested.indent, depth)?;
            }

            count = count.saturating_add(1);
            if count > self.limits.max_sequence_len {
                return Err(YamlError::new(
                    YamlErrorKind::MaxSequenceLengthExceeded,
                    line.offset,
                ));
            }
        }

        if count == 0 {
            return Err(YamlError::new(
                YamlErrorKind::UnexpectedToken,
                self.cursor.position(),
            ));
        }

        Ok(YamlValue::Sequence)
    }

    fn parse_mapping(
        &mut self,
        base_indent: usize,
        depth: usize,
    ) -> Result<YamlValue<'a>, YamlError> {
        let mut count = 0usize;

        loop {
            let Some(line) = self.cursor.peek()? else {
                break;
            };
            if line.indent < base_indent {
                break;
            }
            if line.indent > base_indent {
                return Err(YamlError::new(
                    YamlErrorKind::InvalidIndentation,
                    line.offset,
                ));
            }

            let Some((key, value_part)) = split_mapping_entry(line.content) else {
                break;
            };

            if key.is_empty() {
                return Err(YamlError::new(
                    YamlErrorKind::InvalidMappingKey,
                    line.offset,
                ));
            }
            if key.len() > self.limits.max_scalar_len {
                return Err(YamlError::new(
                    YamlErrorKind::MaxScalarLengthExceeded,
                    line.offset,
                ));
            }

            self.cursor.next()?;

            if !value_part.is_empty() {
                if value_part.len() > self.limits.max_scalar_len {
                    return Err(YamlError::new(
                        YamlErrorKind::MaxScalarLengthExceeded,
                        line.offset,
                    ));
                }
                parse_scalar(value_part, line.offset)?;
            } else {
                let next = self.cursor.peek()?;
                if let Some(next_line) = next {
                    if next_line.indent > base_indent {
                        self.parse_node(next_line.indent, depth)?;
                    }
                }
            }

            count = count.saturating_add(1);
            if count > self.limits.max_mapping_len {
                return Err(YamlError::new(
                    YamlErrorKind::MaxMappingLengthExceeded,
                    line.offset,
                ));
            }
        }

        if count == 0 {
            return Err(YamlError::new(
                YamlErrorKind::UnexpectedToken,
                self.cursor.position(),
            ));
        }

        Ok(YamlValue::Mapping)
    }
}

fn is_sequence_entry(content: &str) -> bool {
    content.starts_with('-') && (content.len() == 1 || content.as_bytes()[1] == b' ')
}

fn has_mapping_separator(content: &str) -> bool {
    split_mapping_entry(content).is_some()
}

fn split_mapping_entry(content: &str) -> Option<(&str, &str)> {
    let bytes = content.as_bytes();
    let mut idx = 0usize;
    while idx < bytes.len() {
        if bytes[idx] == b':' {
            let key = content[..idx].trim();
            let value = content[idx + 1..].trim_start();
            if key.is_empty() {
                return None;
            }
            return Some((key, value));
        }
        idx += 1;
    }
    None
}

pub fn parse_yaml(bytes: &[u8]) -> Result<YamlValue<'_>, YamlError> {
    YamlParser::new(bytes).parse()
}

pub fn parse_yaml_with_max_depth(
    bytes: &[u8],
    max_depth: usize,
) -> Result<YamlValue<'_>, YamlError> {
    YamlParser::new(bytes).with_max_depth(max_depth).parse()
}

pub fn parse_yaml_with_limits(
    bytes: &[u8],
    limits: YamlLimits,
) -> Result<YamlValue<'_>, YamlError> {
    YamlParser::new(bytes).with_limits(limits).parse()
}

pub fn validate_yaml(bytes: &[u8]) -> Result<(), YamlError> {
    YamlParser::new(bytes).validate()
}
