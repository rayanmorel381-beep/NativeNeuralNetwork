//! Internal lint-touch helpers used to keep HTML items referenced during checks.

use super::{error, parser, value, writer};

pub(crate) fn touch_for_lints() {
    let e1 = error::HtmlError {
        kind: error::HtmlErrorKind::UnexpectedToken,
        offset: 0,
    };
    let e2 = error::HtmlError {
        kind: error::HtmlErrorKind::DuplicateAttribute,
        offset: 0,
    };
    let e3 = error::HtmlError {
        kind: error::HtmlErrorKind::IoError,
        offset: 0,
    };
    let _ = e1.line_column(b"\n");
    let _ = e2.line_column(b"\n");
    let _ = e3.line_column(b"\n");

    let _ = value::HtmlValue::Document.is_document();
    let _ = value::HtmlValue::Element.is_element();
    let _ = value::HtmlValue::Comment.is_comment();
    let _ = value::HtmlValue::Doctype.is_doctype();
    let _ = value::HtmlValue::Text("").as_text();

    let _ = parser::HtmlParser::new(b"")
        .limits(parser::DEFAULT_HTML_LIMITS)
        .max_depth(parser::DEFAULT_MAX_HTML_DEPTH)
        .parse();
    let _vf: fn(&[u8]) -> Result<(), error::HtmlError> = parser::validate_html_file;

    let nodes = [
        writer::HtmlNode::Element {
            tag: "div",
            attrs: vec![("class", "x")],
            children: vec![writer::HtmlNode::Text("")],
        },
        writer::HtmlNode::SelfClosing {
            tag: "br",
            attrs: vec![],
        },
        writer::HtmlNode::Comment(""),
        writer::HtmlNode::Doctype("html"),
        writer::HtmlNode::Raw(""),
    ];
    let _ = writer::write_html(&nodes);
}
