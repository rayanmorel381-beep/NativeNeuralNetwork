use core::str;

use super::error::{JsonError, JsonErrorKind};
use super::lexer::Cursor;

pub(crate) fn parse_number(cursor: &mut Cursor<'_>) -> Result<f64, JsonError> {
    let start = cursor.position();

    cursor.try_consume(b'-');

    let first = cursor
        .peek()
        .ok_or(JsonError::new(JsonErrorKind::Eof, cursor.position()))?;

    match first {
        b'0' => {
            cursor.advance(1);
        }
        b'1'..=b'9' => {
            cursor.advance(1);
            while matches!(cursor.peek(), Some(b'0'..=b'9')) {
                cursor.advance(1);
            }
        }
        _ => {
            return Err(JsonError::new(
                JsonErrorKind::InvalidNumber,
                cursor.position(),
            ))
        }
    }

    if cursor.try_consume(b'.') {
        let mut digits = 0usize;
        while matches!(cursor.peek(), Some(b'0'..=b'9')) {
            cursor.advance(1);
            digits += 1;
        }
        if digits == 0 {
            return Err(JsonError::new(
                JsonErrorKind::InvalidNumber,
                cursor.position(),
            ));
        }
    }

    if matches!(cursor.peek(), Some(b'e') | Some(b'E')) {
        cursor.advance(1);
        if matches!(cursor.peek(), Some(b'+') | Some(b'-')) {
            cursor.advance(1);
        }
        let mut digits = 0usize;
        while matches!(cursor.peek(), Some(b'0'..=b'9')) {
            cursor.advance(1);
            digits += 1;
        }
        if digits == 0 {
            return Err(JsonError::new(
                JsonErrorKind::InvalidNumber,
                cursor.position(),
            ));
        }
    }

    let end = cursor.position();
    let s = str::from_utf8(&cursor.bytes()[start..end])
        .map_err(|_| JsonError::new(JsonErrorKind::InvalidUtf8, start))?;
    s.parse::<f64>()
        .map_err(|_| JsonError::new(JsonErrorKind::InvalidNumber, start))
}
