//! INI writer helpers for serializing sections and key/value pairs.

use alloc::{string::String, vec::Vec};

/// Serializes INI sections and key/value entries into text.
pub fn write_ini(sections: &[(&str, Vec<(&str, &str)>)]) -> String {
    let mut out = String::new();
    for (i, (section, entries)) in sections.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        if !section.is_empty() {
            out.push('[');
            out.push_str(section);
            out.push_str("]\n");
        }
        for (key, value) in entries {
            out.push_str(key);
            out.push_str(" = ");
            out.push_str(value);
            out.push('\n');
        }
    }
    out
}

pub fn write_ini_file(
    path: &[u8],
    sections: &[(&str, Vec<(&str, &str)>)],
) -> Result<(), super::error::IniError> {
    let output = write_ini(sections);
    crate::fs::write_file(path, output.as_bytes())
        .map_err(|_| super::error::IniError::new(super::error::IniErrorKind::IoError, 0))
}

pub fn append_ini_file(
    path: &[u8],
    sections: &[(&str, Vec<(&str, &str)>)],
) -> Result<(), super::error::IniError> {
    let output = write_ini(sections);
    crate::fs::append_file(path, output.as_bytes())
        .map_err(|_| super::error::IniError::new(super::error::IniErrorKind::IoError, 0))
}
