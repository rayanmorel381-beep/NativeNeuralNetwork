//! INI parser implementation and section/key validation rules.

use alloc::vec::Vec;

use super::error::{IniError, IniErrorKind};
use super::lexer::LineCursor;
use super::value::IniValue;

/// Resource limits used to bound INI parsing on large or untrusted input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IniLimits {
    /// Maximum number of sections accepted in the document.
    pub max_sections: usize,
    /// Maximum number of keys accepted inside one section.
    pub max_keys_per_section: usize,
    /// Maximum allowed length for a key.
    pub max_key_len: usize,
    /// Maximum allowed length for a value.
    pub max_value_len: usize,
}

/// Conservative default limits for INI parsing and validation.
pub const DEFAULT_INI_LIMITS: IniLimits = IniLimits {
    max_sections: 1024,
    max_keys_per_section: 4096,
    max_key_len: 256,
    max_value_len: 64 * 1024,
};

/// Stateful INI parser configured through [`IniLimits`].
pub struct IniParser<'a> {
    cursor: LineCursor<'a>,
    limits: IniLimits,
}

impl<'a> IniParser<'a> {
    /// Creates a parser with [`DEFAULT_INI_LIMITS`].
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: LineCursor::new(bytes),
            limits: DEFAULT_INI_LIMITS,
        }
    }

    /// Replaces the full parser limit set.
    pub const fn limits(mut self, limits: IniLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Returns the current byte position within the underlying line cursor.
    pub fn position(&self) -> usize {
        self.cursor.position()
    }

    /// Parses the input into a flat sequence of sections and key/value entries.
    pub fn parse(mut self) -> Result<Vec<IniValue<'a>>, IniError> {
        let mut out = Vec::new();
        let mut sections_seen = 0usize;
        let mut keys_in_current = 0usize;

        while let Some(line) = self.cursor.next_line()? {
            if line.content.starts_with('[') {
                let section = parse_section(line.content, line.offset)?;
                if section.is_empty() {
                    return Err(IniError::new(IniErrorKind::EmptySectionName, line.offset));
                }
                sections_seen = sections_seen.saturating_add(1);
                if sections_seen > self.limits.max_sections {
                    return Err(IniError::new(
                        IniErrorKind::MaxSectionsExceeded,
                        line.offset,
                    ));
                }
                keys_in_current = 0;
                out.push(IniValue::Section(section));
            } else {
                let (key, value) = parse_entry(line.content, line.offset)?;
                if key.is_empty() {
                    return Err(IniError::new(IniErrorKind::InvalidKey, line.offset));
                }
                if key.len() > self.limits.max_key_len {
                    return Err(IniError::new(
                        IniErrorKind::MaxKeyLengthExceeded,
                        line.offset,
                    ));
                }
                if value.len() > self.limits.max_value_len {
                    return Err(IniError::new(
                        IniErrorKind::MaxValueLengthExceeded,
                        line.offset,
                    ));
                }
                keys_in_current = keys_in_current.saturating_add(1);
                if keys_in_current > self.limits.max_keys_per_section {
                    return Err(IniError::new(IniErrorKind::MaxKeysExceeded, line.offset));
                }
                out.push(IniValue::Entry { key, value });
            }
        }

        Ok(out)
    }

    /// Validates the INI document without retaining parsed entries.
    pub fn validate(mut self) -> Result<(), IniError> {
        let mut sections_seen = 0usize;
        let mut keys_in_current = 0usize;

        while let Some(line) = self.cursor.next_line()? {
            if line.content.starts_with('[') {
                let section = parse_section(line.content, line.offset)?;
                if section.is_empty() {
                    return Err(IniError::new(IniErrorKind::EmptySectionName, line.offset));
                }
                sections_seen = sections_seen.saturating_add(1);
                if sections_seen > self.limits.max_sections {
                    return Err(IniError::new(
                        IniErrorKind::MaxSectionsExceeded,
                        line.offset,
                    ));
                }
                keys_in_current = 0;
            } else {
                let (key, value) = parse_entry(line.content, line.offset)?;
                if key.is_empty() {
                    return Err(IniError::new(IniErrorKind::InvalidKey, line.offset));
                }
                if key.len() > self.limits.max_key_len {
                    return Err(IniError::new(
                        IniErrorKind::MaxKeyLengthExceeded,
                        line.offset,
                    ));
                }
                if value.len() > self.limits.max_value_len {
                    return Err(IniError::new(
                        IniErrorKind::MaxValueLengthExceeded,
                        line.offset,
                    ));
                }
                keys_in_current = keys_in_current.saturating_add(1);
                if keys_in_current > self.limits.max_keys_per_section {
                    return Err(IniError::new(IniErrorKind::MaxKeysExceeded, line.offset));
                }
            }
        }

        Ok(())
    }
}

fn parse_section(content: &str, offset: usize) -> Result<&str, IniError> {
    let end = content
        .find(']')
        .ok_or(IniError::new(IniErrorKind::UnterminatedSection, offset))?;
    Ok(content[1..end].trim())
}

fn parse_entry(content: &str, offset: usize) -> Result<(&str, &str), IniError> {
    let sep = content
        .find('=')
        .ok_or(IniError::new(IniErrorKind::InvalidKey, offset))?;
    let key = content[..sep].trim();
    let value = content[sep + 1..].trim();
    Ok((key, value))
}

/// Parses INI input using [`DEFAULT_INI_LIMITS`].
pub fn parse_ini(bytes: &[u8]) -> Result<Vec<IniValue<'_>>, IniError> {
    IniParser::new(bytes).parse()
}

/// Performs fast INI validation without retaining the parsed entries.
pub fn validate_ini(bytes: &[u8]) -> Result<(), IniError> {
    IniParser::new(bytes).validate()
}

/// Validates INI input from a byte buffer without using std filesystem APIs.
pub fn validate_ini_file(bytes: &[u8]) -> Result<(), IniError> {
    let mut file_bytes = alloc::vec::Vec::new();
    crate::fs::read_file(bytes, &mut |chunk| {
        file_bytes.extend_from_slice(chunk);
        true
    })
    .map_err(|_| IniError::new(IniErrorKind::IoError, 0))?;
    validate_ini(&file_bytes)
}

pub fn benchmark_validate_ini_file(
    path: &[u8],
    output_dir: &[u8],
) -> Result<nnn_bmk::BenchmarkMetrics<'static>, IniError> {
    crate::fs::ensure_dir(output_dir)
        .map_err(|_| IniError::new(IniErrorKind::IoError, 0))?;
    let started_at = crate::fs::monotonic_ns();
    validate_ini_file(path)?;
    Ok(crate::bmk::parser_metrics(
        "ini",
        "utf-8",
        crate::fs::monotonic_ns().saturating_sub(started_at),
        0,
    ))
}

pub fn benchmark_exit(code: i32) -> ! {
    crate::fs::process_exit(code)
}

/// Performs stricter INI validation, including duplicate-section and duplicate-key checks.
pub fn strict_validate_ini(bytes: &[u8]) -> Result<(), IniError> {
    let values = IniParser::new(bytes).parse()?;
    let mut sections: Vec<&str> = Vec::new();
    let mut current_keys: Vec<&str> = Vec::new();
    for val in &values {
        match val {
            IniValue::Section(name) => {
                if sections.contains(name) {
                    return Err(IniError::new(IniErrorKind::DuplicateSection, 0));
                }
                sections.push(name);
                current_keys.clear();
            }
            IniValue::Entry { key, .. } => {
                if current_keys.contains(key) {
                    return Err(IniError::new(IniErrorKind::DuplicateKey, 0));
                }
                current_keys.push(key);
            }
        }
    }
    Ok(())
}
