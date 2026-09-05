//! TOML serialization helpers and writer-side node definitions.

use alloc::{string::{String, ToString}, vec::Vec};

/// Writer-side TOML node representation used by the serializer.
pub enum TomlNode<'a> {
    String(&'a str),
    Integer(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<TomlNode<'a>>),
    Table(Vec<(&'a str, TomlNode<'a>)>),
}

/// Serializes a TOML table description into textual TOML.
pub fn write_toml(table: &[(&str, TomlNode)]) -> String {
    let mut out = String::new();
    write_table_entries(&mut out, table, &[]);
    out
}

pub fn write_toml_file(
    path: &[u8],
    table: &[(&str, TomlNode)],
) -> Result<(), super::error::TomlError> {
    let output = write_toml(table);
    crate::fs::write_file(path, output.as_bytes())
        .map_err(|_| super::error::TomlError::new(super::error::TomlErrorKind::IoError, 0))
}

pub fn append_toml_file(
    path: &[u8],
    table: &[(&str, TomlNode)],
) -> Result<(), super::error::TomlError> {
    let output = write_toml(table);
    crate::fs::append_file(path, output.as_bytes())
        .map_err(|_| super::error::TomlError::new(super::error::TomlErrorKind::IoError, 0))
}

fn is_array_of_tables(items: &[TomlNode]) -> bool {
    !items.is_empty() && items.iter().all(|i| matches!(i, TomlNode::Table(_)))
}

fn write_header_path(out: &mut String, prefix: &[&str], key: &str) {
    for (i, part) in prefix.iter().enumerate() {
        if i > 0 {
            out.push('.');
        }
        write_key(out, part);
    }
    if !prefix.is_empty() {
        out.push('.');
    }
    write_key(out, key);
}

fn write_table_entries<'a>(
    out: &mut String,
    entries: &[(&'a str, TomlNode<'a>)],
    prefix: &[&'a str],
) {
    let mut subtables: Vec<(&str, &[(&str, TomlNode)])> = Vec::new();
    let mut array_tables: Vec<(&str, &[TomlNode])> = Vec::new();

    for (key, val) in entries {
        match val {
            TomlNode::Table(children) => {
                subtables.push((key, children));
            }
            TomlNode::Array(items) if is_array_of_tables(items) => {
                array_tables.push((key, items));
            }
            _ => {
                write_key(out, key);
                out.push_str(" = ");
                write_value(out, val);
                out.push('\n');
            }
        }
    }

    for (key, children) in &subtables {
        out.push('\n');
        out.push('[');
        write_header_path(out, prefix, key);
        out.push_str("]\n");

        let mut new_prefix = prefix.to_vec();
        new_prefix.push(key);
        write_table_entries(out, children, &new_prefix);
    }

    for (key, items) in &array_tables {
        for item in *items {
            if let TomlNode::Table(children) = item {
                out.push('\n');
                out.push_str("[[");
                write_header_path(out, prefix, key);
                out.push_str("]]\n");

                let mut new_prefix = prefix.to_vec();
                new_prefix.push(key);
                write_table_entries(out, children, &new_prefix);
            }
        }
    }
}

fn write_value(out: &mut String, node: &TomlNode) {
    match node {
        TomlNode::String(s) => {
            out.push('"');
            for c in s.chars() {
                match c {
                    '"' => out.push_str("\\\""),
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c => out.push(c),
                }
            }
            out.push('"');
        }
        TomlNode::Integer(n) => out.push_str(&n.to_string()),
        TomlNode::Float(f) => {
            if f.is_finite() {
                out.push_str(&alloc::format!("{f}"));
            } else if f.is_nan() {
                out.push_str("nan");
            } else if *f > 0.0 {
                out.push_str("inf");
            } else {
                out.push_str("-inf");
            }
        }
        TomlNode::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        TomlNode::Array(items) => {
            out.push('[');
            let mut first = true;
            for item in items {
                if !first {
                    out.push_str(", ");
                }
                write_value(out, item);
                first = false;
            }
            out.push(']');
        }
        TomlNode::Table(pairs) => {
            out.push('{');
            let mut first = true;
            for (k, v) in pairs {
                if !first {
                    out.push_str(", ");
                }
                write_key(out, k);
                out.push_str(" = ");
                write_value(out, v);
                first = false;
            }
            out.push('}');
        }
    }
}

fn write_key(out: &mut String, key: &str) {
    let needs_quotes = key.is_empty()
        || key
            .chars()
            .any(|c| !c.is_ascii_alphanumeric() && c != '-' && c != '_');
    if needs_quotes {
        out.push('"');
        out.push_str(key);
        out.push('"');
    } else {
        out.push_str(key);
    }
}
