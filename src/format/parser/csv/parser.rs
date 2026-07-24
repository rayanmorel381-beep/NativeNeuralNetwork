use super::error::{CsvError, CsvErrorKind};
use super::lexer::Cursor;
use super::value::CsvValue;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsvLimits {
    pub max_rows: usize,
    pub max_columns: usize,
    pub max_field_len: usize,
    pub max_node_count: usize,
}

pub const DEFAULT_CSV_LIMITS: CsvLimits = CsvLimits {
    max_rows: 1_000_000,
    max_columns: 16_384,
    max_field_len: 64 * 1024,
    max_node_count: 2_000_000,
};

pub struct CsvParser<'a> {
    cursor: Cursor<'a>,
    limits: CsvLimits,
    rows_seen: usize,
    nodes_seen: usize,
}

impl<'a> CsvParser<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            cursor: Cursor::new(bytes),
            limits: DEFAULT_CSV_LIMITS,
            rows_seen: 0,
            nodes_seen: 0,
        }
    }

    pub const fn with_limits(mut self, limits: CsvLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn parse(mut self) -> Result<CsvValue<'a>, CsvError> {
        self.parse_all()?;
        Ok(CsvValue::Table)
    }

    pub fn validate(mut self) -> Result<(), CsvError> {
        self.parse_all()
    }

    fn parse_all(&mut self) -> Result<(), CsvError> {
        if self.cursor.is_eof() {
            return Ok(());
        }

        loop {
            self.parse_row()?;
            if self.cursor.is_eof() {
                return Ok(());
            }
        }
    }

    fn parse_row(&mut self) -> Result<(), CsvError> {
        self.rows_seen = self.rows_seen.saturating_add(1);
        if self.rows_seen > self.limits.max_rows {
            return Err(CsvError::new(
                CsvErrorKind::MaxRowsExceeded,
                self.cursor.position(),
            ));
        }

        let mut cols = 0usize;

        loop {
            self.parse_field()?;
            cols = cols.saturating_add(1);
            if cols > self.limits.max_columns {
                return Err(CsvError::new(
                    CsvErrorKind::MaxColumnsExceeded,
                    self.cursor.position(),
                ));
            }

            match self.cursor.peek() {
                Some(b',') => {
                    self.cursor.advance(1);
                    continue;
                }
                Some(b'\n') => {
                    self.cursor.advance(1);
                    break;
                }
                Some(b'\r') => {
                    self.cursor.advance(1);
                    if self.cursor.peek() == Some(b'\n') {
                        self.cursor.advance(1);
                    }
                    break;
                }
                None => break,
                _ => {
                    return Err(CsvError::new(
                        CsvErrorKind::TrailingCharactersAfterQuote,
                        self.cursor.position(),
                    ));
                }
            }
        }

        Ok(())
    }

    fn parse_field(&mut self) -> Result<(), CsvError> {
        self.nodes_seen = self.nodes_seen.saturating_add(1);
        if self.nodes_seen > self.limits.max_node_count {
            return Err(CsvError::new(
                CsvErrorKind::MaxNodeCountExceeded,
                self.cursor.position(),
            ));
        }

        match self.cursor.peek() {
            Some(b'"') => self.parse_quoted_field(),
            _ => self.parse_unquoted_field(),
        }
    }

    fn parse_unquoted_field(&mut self) -> Result<(), CsvError> {
        let start = self.cursor.position();

        while let Some(b) = self.cursor.peek() {
            if b == b',' || b == b'\n' || b == b'\r' {
                break;
            }
            if b == b'"' {
                return Err(CsvError::new(
                    CsvErrorKind::UnexpectedQuote,
                    self.cursor.position(),
                ));
            }
            self.cursor.advance(1);
        }

        let end = self.cursor.position();
        let len = end.saturating_sub(start);
        if len > self.limits.max_field_len {
            return Err(CsvError::new(CsvErrorKind::MaxFieldLengthExceeded, start));
        }

        core::str::from_utf8(&self.cursor.bytes()[start..end])
            .map_err(|_| CsvError::new(CsvErrorKind::InvalidUtf8, start))?;

        Ok(())
    }

    fn parse_quoted_field(&mut self) -> Result<(), CsvError> {
        let quote_start = self.cursor.position();
        self.cursor.next();
        let content_start = self.cursor.position();

        loop {
            let b = self.cursor.next().ok_or(CsvError::new(
                CsvErrorKind::UnterminatedQuotedField,
                quote_start,
            ))?;

            if b == b'"' {
                if self.cursor.peek() == Some(b'"') {
                    self.cursor.advance(1);
                    continue;
                }
                break;
            }
        }

        let content_end = self.cursor.position().saturating_sub(1);
        let len = content_end.saturating_sub(content_start);
        if len > self.limits.max_field_len {
            return Err(CsvError::new(
                CsvErrorKind::MaxFieldLengthExceeded,
                content_start,
            ));
        }

        core::str::from_utf8(&self.cursor.bytes()[content_start..content_end])
            .map_err(|_| CsvError::new(CsvErrorKind::InvalidUtf8, content_start))?;

        match self.cursor.peek() {
            Some(b',') | Some(b'\n') | Some(b'\r') | None => Ok(()),
            Some(_) => Err(CsvError::new(
                CsvErrorKind::TrailingCharactersAfterQuote,
                self.cursor.position(),
            )),
        }
    }
}

pub fn parse_csv(bytes: &[u8]) -> Result<CsvValue<'_>, CsvError> {
    CsvParser::new(bytes).parse()
}

pub fn parse_csv_with_limits(bytes: &[u8], limits: CsvLimits) -> Result<CsvValue<'_>, CsvError> {
    CsvParser::new(bytes).with_limits(limits).parse()
}

pub fn validate_csv(bytes: &[u8]) -> Result<(), CsvError> {
    CsvParser::new(bytes).validate()
}
