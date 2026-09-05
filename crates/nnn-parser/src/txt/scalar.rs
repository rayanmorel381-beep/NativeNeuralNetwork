pub fn normalize_text<'a>(bytes: &'a [u8]) -> Result<&'a str, &'static str> {
    core::str::from_utf8(bytes).map_err(|_| "invalid UTF-8 in text payload")
}

pub fn count_lines(text: &str) -> usize {
    if text.is_empty() {
        return 0;
    }
    text.lines().count()
}
