//! INI value representation used by the format-specific parser.

/// Format-specific INI value representation used by the parser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum IniValue<'a> {
    /// Borrowed name of a section header.
    Section(&'a str),
    /// Borrowed key/value entry from within a section.
    Entry {
        /// Borrowed key name.
        key: &'a str,
        /// Borrowed raw value text.
        value: &'a str,
    },
}

impl<'a> IniValue<'a> {
    /// Borrows the section name when the value is `Section`.
    pub const fn as_section(&self) -> Option<&str> {
        match self {
            IniValue::Section(s) => Some(s),
            _ => None,
        }
    }

    /// Borrows the `(key, value)` pair when the value is `Entry`.
    pub const fn as_entry(&self) -> Option<(&str, &str)> {
        match self {
            IniValue::Entry { key, value } => Some((key, value)),
            _ => None,
        }
    }

    /// Returns `true` when the value is `Section`.
    pub const fn is_section(&self) -> bool {
        matches!(self, IniValue::Section(_))
    }

    /// Returns `true` when the value is an `Entry` pair.
    pub const fn is_entry(&self) -> bool {
        matches!(self, IniValue::Entry { .. })
    }
}
