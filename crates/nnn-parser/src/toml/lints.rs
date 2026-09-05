//! Internal lint-touch helpers used to keep TOML items referenced during checks.

use super::{error, parser, value, writer};

pub(crate) fn touch_for_lints() {
    let e1 = error::TomlError {
        kind: error::TomlErrorKind::InvalidDate,
        offset: 0,
    };
    let e2 = error::TomlError {
        kind: error::TomlErrorKind::UnterminatedInlineTable,
        offset: 0,
    };
    let e3 = error::TomlError {
        kind: error::TomlErrorKind::DuplicateKey,
        offset: 0,
    };
    let e4 = error::TomlError {
        kind: error::TomlErrorKind::DuplicateTable,
        offset: 0,
    };
    let e5 = error::TomlError {
        kind: error::TomlErrorKind::IoError,
        offset: 0,
    };
    let _ = e1.line_column(b"\n");
    let _ = e2.line_column(b"\n");
    let _ = e3.line_column(b"\n");
    let _ = e4.line_column(b"\n");
    let _ = e5.line_column(b"\n");

    let _ = value::TomlValue::String("").as_str();
    let _ = value::TomlValue::Integer(1).as_integer();
    let _ = value::TomlValue::Float(1.0).as_float();
    let _ = value::TomlValue::Bool(true).as_bool();
    let _ = value::TomlValue::Array.is_array();
    let _ = value::TomlValue::Table.is_table();

    let _ = parser::TomlParser::new(b"")
        .limits(parser::DEFAULT_TOML_LIMITS)
        .parse();
    let _vf: fn(&[u8]) -> Result<(), error::TomlError> = parser::validate_toml_file;

    let table: [(&str, writer::TomlNode); 6] = [
        ("s", writer::TomlNode::String("")),
        ("i", writer::TomlNode::Integer(1)),
        ("f", writer::TomlNode::Float(1.0)),
        ("b", writer::TomlNode::Bool(true)),
        ("a", writer::TomlNode::Array(vec![writer::TomlNode::Integer(2)])),
        (
            "t",
            writer::TomlNode::Table(vec![("k", writer::TomlNode::String("v"))]),
        ),
    ];
    let _ = writer::write_toml(&table);
}
