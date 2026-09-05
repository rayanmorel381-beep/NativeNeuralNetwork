//! Internal lint-touch helpers used to keep INI items referenced during checks.

use super::{error, lexer, parser, value, writer};

pub(crate) fn touch_for_lints() {
    let e1 = error::IniError {
        kind: error::IniErrorKind::DuplicateSection,
        offset: 0,
    };
    let e2 = error::IniError {
        kind: error::IniErrorKind::DuplicateKey,
        offset: 0,
    };
    let e3 = error::IniError {
        kind: error::IniErrorKind::IoError,
        offset: 0,
    };
    let _ = e1.line_column(b"\n");
    let _ = e2.line_column(b"\n");
    let _ = e3.line_column(b"\n");

    let sec = value::IniValue::Section("s");
    let ent = value::IniValue::Entry { key: "k", value: "v" };
    let _ = sec.as_section();
    let _ = sec.is_section();
    let _ = ent.as_entry();
    let _ = ent.is_entry();

    let cur = lexer::LineCursor::new(b"");
    let _ = cur.position();

    let _ = parser::IniParser::new(b"").limits(parser::DEFAULT_INI_LIMITS);
    let _ = parser::IniParser::new(b"").position();
    let _ = parser::IniParser::new(b"")
        .limits(parser::DEFAULT_INI_LIMITS)
        .parse();
    let _vf: fn(&[u8]) -> Result<(), error::IniError> = parser::validate_ini_file;

    let _ = writer::write_ini(&[]);
}
