use super::error::{YamlError, YamlErrorKind};
use super::value::YamlValue;

pub fn parse_scalar<'a>(raw: &'a str, offset: usize) -> Result<YamlValue<'a>, YamlError> {
    let text = raw.trim();
    if text.is_empty() {
        return Ok(YamlValue::Null);
    }

    if has_unsupported_feature_prefix(text) {
        return Err(YamlError::new(YamlErrorKind::UnsupportedFeature, offset));
    }

    if text == "null" || text == "Null" || text == "NULL" || text == "~" {
        return Ok(YamlValue::Null);
    }

    if text == "true" || text == "True" || text == "TRUE" {
        return Ok(YamlValue::Bool(true));
    }

    if text == "false" || text == "False" || text == "FALSE" {
        return Ok(YamlValue::Bool(false));
    }

    if let Some(inner) = strip_quoted(text) {
        return Ok(YamlValue::String(inner));
    }

    if looks_like_number(text) {
        let number = text
            .parse::<f64>()
            .map_err(|_| YamlError::new(YamlErrorKind::InvalidScalar, offset))?;
        return Ok(YamlValue::Number(number));
    }

    Ok(YamlValue::String(text))
}

fn strip_quoted(text: &str) -> Option<&str> {
    if text.len() >= 2 {
        let b = text.as_bytes();
        if (b[0] == b'"' && b[text.len() - 1] == b'"')
            || (b[0] == b'\'' && b[text.len() - 1] == b'\'')
        {
            return Some(&text[1..text.len() - 1]);
        }
    }
    None
}

fn has_unsupported_feature_prefix(text: &str) -> bool {
    text.starts_with('&')
        || text.starts_with('*')
        || text.starts_with('!')
        || text.starts_with('|')
        || text.starts_with('>')
        || text.starts_with('[')
        || text.starts_with('{')
}

fn looks_like_number(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return false;
    }

    let mut idx = 0usize;
    if bytes[0] == b'+' || bytes[0] == b'-' {
        idx = 1;
    }
    if idx >= bytes.len() {
        return false;
    }

    let mut has_digit = false;
    while idx < bytes.len() {
        let b = bytes[idx];
        if b.is_ascii_digit() {
            has_digit = true;
            idx += 1;
            continue;
        }
        if b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
            idx += 1;
            continue;
        }
        return false;
    }

    has_digit
}
