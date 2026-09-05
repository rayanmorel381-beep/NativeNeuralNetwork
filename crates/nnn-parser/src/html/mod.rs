//! HTML support for Omnifile.
//!
//! This module provides a lightweight structural HTML parser and renderer used
//! by the format-neutral core facade. It is designed for safe validation,
//! conversion, and preview generation — not to emulate a browser engine, CSS
//! cascade, or full DOM implementation.
//!
//! ## What it covers
//!
//! - elements, text nodes, comments, and doctypes
//! - attribute parsing and basic entity validation
//! - strict checks for malformed or mismatched markup
//! - serialization back into normalized HTML
//!
//! ## How the code is organized
//!
//! - `parser` performs the high-level document walk and structural checks
//! - `lexer` offers the byte cursor and tag-scanning primitives
//! - `entity` validates HTML character references such as `&amp;`
//! - `value` defines the native `HtmlValue` representation
//! - `writer` turns structured data back into HTML or preview pages
//! - `error` contains HTML-specific error kinds and positions
//!
//! If you only need to parse or convert HTML from user code, `omnifile::core`
//! is usually the simpler entry point.

mod entity;
pub mod error;
mod lexer;
pub mod value;

pub mod parser;
pub mod writer;

pub use error::HtmlErrorKind;
pub use parser::{
	benchmark_exit, benchmark_validate_html_file, parse_html, validate_html, validate_html_file,
};
pub use writer::{append_html_file, write_html_file};
pub use value::HtmlValue;
