//! TOML value representation used by the format-specific parser.

/// Format-specific TOML value representation used by the parser.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TomlValue<'a> {
    /// Borrowed TOML string value.
    String(&'a str),
    /// Integer value.
    Integer(i64),
    /// Floating-point value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Marker for an array container.
    Array,
    /// Marker for a standard table.
    Table,
    /// Marker for an inline table.
    InlineTable,
}

impl<'a> TomlValue<'a> {
    /// Borrows the underlying string when the value is `String`.
    pub const fn as_str(&self) -> Option<&str> {
        match self {
            TomlValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the contained integer when the value is `Integer`.
    pub const fn as_integer(&self) -> Option<i64> {
        match self {
            TomlValue::Integer(n) => Some(*n),
            _ => None,
        }
    }

    /// Returns the contained floating-point number when the value is `Float`.
    pub const fn as_float(&self) -> Option<f64> {
        match self {
            TomlValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the contained boolean when the value is `Bool`.
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            TomlValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns `true` when the value is an array container marker.
    pub const fn is_array(&self) -> bool {
        matches!(self, TomlValue::Array)
    }

    /// Returns `true` when the value is either a normal or inline table marker.
    pub const fn is_table(&self) -> bool {
        matches!(self, TomlValue::Table | TomlValue::InlineTable)
    }
}
