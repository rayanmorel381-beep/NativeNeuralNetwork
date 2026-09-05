use core::char;
use core::str;

use super::error::{JsonError, JsonErrorKind};
use super::lexer::Cursor;

pub(crate) struct ParsedString<'a> {
    pub raw: &'a str,
    pub start: usize,
    pub end: usize,
}

pub(crate) fn parse_string<'a>(cursor: &mut Cursor<'a>) -> Result<ParsedString<'a>, JsonError> {
    cursor.consume(b'"')?;

    let start = cursor.position();
    let mut seg_start = start;

    loop {
        let b = cursor
            .peek()
            .ok_or(JsonError::new(JsonErrorKind::Eof, cursor.position()))?;

        if b == b'"' {
            validate_utf8_segment(cursor, seg_start, cursor.position())?;
            let end = cursor.position();
            cursor.advance(1);
            let raw = str::from_utf8(&cursor.bytes()[start..end])
                .map_err(|_| JsonError::new(JsonErrorKind::InvalidUtf8, start))?;
            return Ok(ParsedString { raw, start, end });
        }

        if b == b'\\' {
            validate_utf8_segment(cursor, seg_start, cursor.position())?;
            cursor.advance(1);
            parse_escape(cursor)?;
            seg_start = cursor.position();
            continue;
        }

        if b < 0x20 {
            return Err(JsonError::new(
                JsonErrorKind::ControlCharacterInString,
                cursor.position(),
            ));
        }

        cursor.advance(1);
    }
}

fn parse_escape(cursor: &mut Cursor<'_>) -> Result<(), JsonError> {
    let esc = cursor
        .peek()
        .ok_or(JsonError::new(JsonErrorKind::Eof, cursor.position()))?;
    cursor.advance(1);

    match esc {
        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {}
        b'u' => {
            let first = parse_u16_hex(cursor)?;
            if (0xD800..=0xDBFF).contains(&first) {
                cursor.consume(b'\\')?;
                cursor.consume(b'u')?;
                let second = parse_u16_hex(cursor)?;
                if !(0xDC00..=0xDFFF).contains(&second) {
                    return Err(JsonError::new(
                        JsonErrorKind::InvalidUnicodeSurrogate,
                        cursor.position(),
                    ));
                }
                let code =
                    0x10000u32 + (((first as u32) - 0xD800) << 10) + ((second as u32) - 0xDC00);
                char::from_u32(code).ok_or(JsonError::new(
                    JsonErrorKind::InvalidUnicodeEscape,
                    cursor.position(),
                ))?;
            } else if (0xDC00..=0xDFFF).contains(&first) {
                return Err(JsonError::new(
                    JsonErrorKind::InvalidUnicodeSurrogate,
                    cursor.position(),
                ));
            } else {
                char::from_u32(first as u32).ok_or(JsonError::new(
                    JsonErrorKind::InvalidUnicodeEscape,
                    cursor.position(),
                ))?;
            }
        }
        _ => {
            return Err(JsonError::new(
                JsonErrorKind::InvalidEscape,
                cursor.position() - 1,
            ))
        }
    }

    Ok(())
}

fn parse_u16_hex(cursor: &mut Cursor<'_>) -> Result<u16, JsonError> {
    if cursor.position() + 4 > cursor.len() {
        return Err(JsonError::new(JsonErrorKind::Eof, cursor.position()));
    }
    let mut v = 0u16;
    for _ in 0..4 {
        let b = cursor
            .peek()
            .ok_or(JsonError::new(JsonErrorKind::Eof, cursor.position()))?;
        cursor.advance(1);
        let d = match b {
            b'0'..=b'9' => (b - b'0') as u16,
            b'a'..=b'f' => (b - b'a' + 10) as u16,
            b'A'..=b'F' => (b - b'A' + 10) as u16,
            _ => {
                return Err(JsonError::new(
                    JsonErrorKind::InvalidUnicodeEscape,
                    cursor.position() - 1,
                ));
            }
        };
        v = (v << 4) | d;
    }
    Ok(v)
}

fn validate_utf8_segment(cursor: &Cursor<'_>, start: usize, end: usize) -> Result<(), JsonError> {
    if start == end {
        return Ok(());
    }
    str::from_utf8(&cursor.bytes()[start..end])
        .map_err(|_| JsonError::new(JsonErrorKind::InvalidUtf8, start))?;
    Ok(())
}
