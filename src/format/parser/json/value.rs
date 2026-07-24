#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonValue<'a> {
    Null,
    Bool(bool),
    Number(f64),
    String(&'a str),
    Array,
    Object,
}

impl<'a> JsonValue<'a> {
    pub const fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object)
    }

    pub const fn is_array(&self) -> bool {
        matches!(self, JsonValue::Array)
    }

    pub const fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::String(v) => Some(v),
            _ => None,
        }
    }

    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            JsonValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub const fn as_number(&self) -> Option<f64> {
        match self {
            JsonValue::Number(v) => Some(*v),
            _ => None,
        }
    }
}
