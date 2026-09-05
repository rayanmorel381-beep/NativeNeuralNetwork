//! Internal lint-touch helpers used to keep XML items referenced during checks.

use super::{error, parser, value, writer};

pub(crate) fn touch_for_lints() {
    let e1 = error::XmlError {
        kind: error::XmlErrorKind::DuplicateAttribute,
        offset: 0,
    };
    let e2 = error::XmlError {
        kind: error::XmlErrorKind::InvalidEntity,
        offset: 0,
    };
    let e3 = error::XmlError {
        kind: error::XmlErrorKind::InvalidDeclaration,
        offset: 0,
    };
    let e4 = error::XmlError {
        kind: error::XmlErrorKind::IoError,
        offset: 0,
    };
    let _ = e1.line_column(b"\n");
    let _ = e2.line_column(b"\n");
    let _ = e3.line_column(b"\n");
    let _ = e4.line_column(b"\n");

    let _ = value::XmlValue::Document.is_document();
    let _ = value::XmlValue::Element.is_element();
    let _ = value::XmlValue::Comment.is_comment();
    let _ = value::XmlValue::Text("").as_text();
    let _ = value::XmlValue::Cdata("").as_cdata();
    let _ = value::XmlValue::ProcessingInstruction;

    let _ = parser::XmlParser::new(b"")
        .limits(parser::DEFAULT_XML_LIMITS)
        .max_depth(parser::DEFAULT_XML_LIMITS.max_depth)
        .parse();
    let _vf: fn(&[u8]) -> Result<(), error::XmlError> = parser::validate_xml_file;

    let nodes = [
        writer::XmlNode::Element {
            tag: "root",
            attrs: vec![("id", "1")],
            children: vec![writer::XmlNode::Text("")],
        },
        writer::XmlNode::SelfClosing {
            tag: "br",
            attrs: vec![],
        },
        writer::XmlNode::Comment(""),
        writer::XmlNode::Cdata(""),
        writer::XmlNode::ProcessingInstruction {
            target: "xml-stylesheet",
            content: "",
        },
        writer::XmlNode::Raw(""),
    ];
    let _ = writer::write_xml(&nodes, false);
}
