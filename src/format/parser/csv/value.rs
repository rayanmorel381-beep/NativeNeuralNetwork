#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CsvValue<'a> {
    Table,
    Record,
    Field(&'a str),
}

impl<'a> CsvValue<'a> {
    pub const fn is_table(&self) -> bool {
        matches!(self, CsvValue::Table)
    }

    pub const fn is_record(&self) -> bool {
        matches!(self, CsvValue::Record)
    }

    pub const fn as_field(&self) -> Option<&str> {
        match self {
            CsvValue::Field(v) => Some(v),
            _ => None,
        }
    }
}
