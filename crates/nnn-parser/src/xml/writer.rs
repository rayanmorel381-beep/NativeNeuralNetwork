//! XML serialization helpers and writer-side node definitions.

use alloc::{string::String, vec::Vec};

/// Writer-side XML node representation used by the serializer.
pub enum XmlNode<'a> {
    Element {
        tag: &'a str,
        attrs: Vec<(&'a str, &'a str)>,
        children: Vec<XmlNode<'a>>,
    },
    SelfClosing {
        tag: &'a str,
        attrs: Vec<(&'a str, &'a str)>,
    },
    Text(&'a str),
    Comment(&'a str),
    Cdata(&'a str),
    ProcessingInstruction {
        target: &'a str,
        content: &'a str,
    },
    Raw(&'a str),
}

/// Serializes XML writer nodes into an XML document or fragment.
pub fn write_xml(nodes: &[XmlNode], declaration: bool) -> String {
    let mut out = String::new();
    if declaration {
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    }
    for node in nodes {
        write_node(&mut out, node);
    }
    out
}

pub fn write_xml_file(
    path: &[u8],
    nodes: &[XmlNode],
    declaration: bool,
) -> Result<(), super::error::XmlError> {
    let output = write_xml(nodes, declaration);
    crate::fs::write_file(path, output.as_bytes())
        .map_err(|_| super::error::XmlError::new(super::error::XmlErrorKind::IoError, 0))
}

pub fn append_xml_file(
    path: &[u8],
    nodes: &[XmlNode],
    declaration: bool,
) -> Result<(), super::error::XmlError> {
    let output = write_xml(nodes, declaration);
    crate::fs::append_file(path, output.as_bytes())
        .map_err(|_| super::error::XmlError::new(super::error::XmlErrorKind::IoError, 0))
}

fn write_node(out: &mut String, node: &XmlNode) {
    match node {
        XmlNode::Element {
            tag,
            attrs,
            children,
        } => {
            out.push('<');
            out.push_str(tag);
            write_attrs(out, attrs);
            if children.is_empty() {
                out.push_str(" />");
            } else {
                out.push('>');
                for child in children {
                    write_node(out, child);
                }
                out.push_str("</");
                out.push_str(tag);
                out.push('>');
            }
        }
        XmlNode::SelfClosing { tag, attrs } => {
            out.push('<');
            out.push_str(tag);
            write_attrs(out, attrs);
            out.push_str(" />");
        }
        XmlNode::Text(t) => escape_text(out, t),
        XmlNode::Comment(c) => {
            out.push_str("<!--");
            out.push_str(c);
            out.push_str("-->");
        }
        XmlNode::Cdata(c) => {
            out.push_str("<![CDATA[");
            out.push_str(c);
            out.push_str("]]>");
        }
        XmlNode::ProcessingInstruction { target, content } => {
            out.push_str("<?");
            out.push_str(target);
            if !content.is_empty() {
                out.push(' ');
                out.push_str(content);
            }
            out.push_str("?>");
        }
        XmlNode::Raw(r) => out.push_str(r),
    }
}

fn write_attrs(out: &mut String, attrs: &[(&str, &str)]) {
    for (k, v) in attrs {
        out.push(' ');
        out.push_str(k);
        out.push_str("=\"");
        escape_attr(out, v);
        out.push('"');
    }
}

fn escape_text(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            c => out.push(c),
        }
    }
}

fn escape_attr(out: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            c => out.push(c),
        }
    }
}
