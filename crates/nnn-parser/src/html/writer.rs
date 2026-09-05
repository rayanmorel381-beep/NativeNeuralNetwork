//! HTML serialization helpers for the parser crate.

use alloc::{string::String, vec, vec::Vec};

/// Writer-side HTML node representation used by the serializer.
pub enum HtmlNode<'a> {
    Element {
        tag: &'a str,
        attrs: Vec<(&'a str, &'a str)>,
        children: Vec<HtmlNode<'a>>,
    },
    SelfClosing {
        tag: &'a str,
        attrs: Vec<(&'a str, &'a str)>,
    },
    Text(&'a str),
    Comment(&'a str),
    Doctype(&'a str),
    Raw(&'a str),
}

/// Serializes HTML writer nodes into a textual document fragment.
pub fn write_html(nodes: &[HtmlNode<'_>]) -> String {
    let mut out = String::new();
    for node in nodes {
        write_node(&mut out, node);
    }
    out
}

pub fn write_html_file(path: &[u8], nodes: &[HtmlNode<'_>]) -> Result<(), super::error::HtmlError> {
    let output = write_html(nodes);
    crate::fs::write_file(path, output.as_bytes())
        .map_err(|_| super::error::HtmlError::new(super::error::HtmlErrorKind::IoError, 0))
}

pub fn append_html_file(path: &[u8], nodes: &[HtmlNode<'_>]) -> Result<(), super::error::HtmlError> {
    let output = write_html(nodes);
    crate::fs::append_file(path, output.as_bytes())
        .map_err(|_| super::error::HtmlError::new(super::error::HtmlErrorKind::IoError, 0))
}

fn write_node(out: &mut String, node: &HtmlNode<'_>) {
    match node {
        HtmlNode::Element {
            tag,
            attrs,
            children,
        } => {
            out.push('<');
            out.push_str(tag);
            for (k, v) in attrs {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                escape_attr(out, v);
                out.push('"');
            }
            out.push('>');
            for child in children {
                write_node(out, child);
            }
            out.push_str("</");
            out.push_str(tag);
            out.push('>');
        }
        HtmlNode::SelfClosing { tag, attrs } => {
            out.push('<');
            out.push_str(tag);
            for (k, v) in attrs {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                escape_attr(out, v);
                out.push('"');
            }
            out.push_str(" />");
        }
        HtmlNode::Text(t) => escape_text(out, t),
        HtmlNode::Comment(c) => {
            out.push_str("<!--");
            out.push_str(c);
            out.push_str("-->");
        }
        HtmlNode::Doctype(d) => {
            out.push_str("<!DOCTYPE ");
            out.push_str(d);
            out.push('>');
        }
        HtmlNode::Raw(r) => out.push_str(r),
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

/// Minimal preview helper for HTML writer consumers.
pub fn render_omni_page(html: &str) -> String {
    let nodes = [
        HtmlNode::Doctype("html"),
        HtmlNode::Element {
            tag: "html",
            attrs: vec![],
            children: vec![HtmlNode::Element {
                tag: "body",
                attrs: vec![],
                children: vec![HtmlNode::Raw(html)],
            }],
        },
    ];
    write_html(&nodes)
}
