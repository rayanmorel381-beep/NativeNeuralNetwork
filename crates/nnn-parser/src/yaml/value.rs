#[derive(Clone, Copy, Debug, PartialEq)]
pub enum YamlValue<'a> {
    Null,
    Bool(bool),
    Number(f64),
    String(&'a str),
    Sequence,
    Mapping,
}

impl<'a> YamlValue<'a> {
    pub const fn is_mapping(&self) -> bool {
        matches!(self, YamlValue::Mapping)
    }

    pub const fn is_sequence(&self) -> bool {
        matches!(self, YamlValue::Sequence)
    }

    pub const fn as_str(&self) -> Option<&str> {
        match self {
            YamlValue::String(v) => Some(v),
            _ => None,
        }
    }

    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            YamlValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub const fn as_number(&self) -> Option<f64> {
        match self {
            YamlValue::Number(v) => Some(*v),
            _ => None,
        }
    }
}
