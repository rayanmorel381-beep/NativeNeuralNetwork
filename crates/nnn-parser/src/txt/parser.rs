use super::{
    error::{TxtError, TxtErrorKind},
    value::TxtValue,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TxtLimits {
    pub max_text_len: usize,
    pub max_lines: usize,
    pub max_line_len: usize,
}

pub const DEFAULT_TXT_LIMITS: TxtLimits = TxtLimits {
    max_text_len: 1 << 20,
    max_lines: 100_000,
    max_line_len: 64 * 1024,
};

pub struct TxtParser<'a> {
    bytes: &'a [u8],
    limits: TxtLimits,
}

impl<'a> TxtParser<'a> {
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            limits: DEFAULT_TXT_LIMITS,
        }
    }

    pub const fn with_limits(mut self, limits: TxtLimits) -> Self {
        self.limits = limits;
        self
    }

    pub fn parse(self) -> Result<TxtValue<'a>, TxtError> {
        let text = super::scalar::normalize_text(self.bytes)
            .map_err(|_| TxtError::new(TxtErrorKind::InvalidUtf8, self.bytes.len()))?;

        if text.len() > self.limits.max_text_len {
            return Err(TxtError::new(TxtErrorKind::MaxTextLengthExceeded, text.len()));
        }

        let lines = super::scalar::count_lines(text);
        if lines > self.limits.max_lines {
            return Err(TxtError::new(TxtErrorKind::MaxLinesExceeded, text.len()));
        }

        for line in text.lines() {
            if line.len() > self.limits.max_line_len {
                return Err(TxtError::new(
                    TxtErrorKind::MaxLineLengthExceeded,
                    text.len().saturating_sub(line.len()),
                ));
            }
        }

        Ok(TxtValue::Text(text))
    }

    pub fn validate(self) -> Result<(), TxtError> {
        let _ = self.parse()?;
        Ok(())
    }
}

pub fn parse_txt(bytes: &[u8]) -> Result<TxtValue<'_>, TxtError> {
    TxtParser::new(bytes).parse()
}

pub fn parse_txt_with_limits(bytes: &[u8], limits: TxtLimits) -> Result<TxtValue<'_>, TxtError> {
    TxtParser::new(bytes).with_limits(limits).parse()
}

pub fn validate_txt(bytes: &[u8]) -> Result<(), TxtError> {
    TxtParser::new(bytes).validate()
}

pub fn validate_txt_file(path: &[u8]) -> Result<(), TxtError> {
    let mut bytes = alloc::vec::Vec::new();
    crate::fs::read_file(path, &mut |chunk| {
        bytes.extend_from_slice(chunk);
        true
    })
    .map_err(|_| TxtError::new(TxtErrorKind::IoError, 0))?;
    validate_txt(&bytes)
}

pub fn benchmark_validate_txt_file(
    path: &[u8],
    output_dir: &[u8],
) -> Result<nnn_bmk::BenchmarkMetrics<'static>, TxtError> {
    crate::fs::ensure_dir(output_dir)
        .map_err(|_| TxtError::new(TxtErrorKind::IoError, 0))?;
    let started_at = crate::fs::monotonic_ns();
    validate_txt_file(path)?;
    Ok(crate::bmk::parser_metrics(
        "txt",
        "utf-8",
        crate::fs::monotonic_ns().saturating_sub(started_at),
        0,
    ))
}

pub fn benchmark_exit(code: i32) -> ! {
    crate::fs::process_exit(code)
}
