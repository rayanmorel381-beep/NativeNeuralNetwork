//! XML support for Omnifile.
//!
//! This module implements a small structural XML parser and serializer used by
//! the core conversion API. It focuses on element trees, attributes, comments,
//! CDATA, and well-formedness checks rather than on the full XML ecosystem.
//!
//! ## Core goals
//!
//! - validate XML structure and tag matching
//! - expose a native `XmlValue` representation
//! - convert element trees through `OmniValue`
//! - serialize normalized XML text
//! - provide strict checks for malformed comment or multi-root cases
//!
//! ## Submodules
//!
//! - `parser` performs the document-level parse
//! - `lexer` provides low-level scanning utilities
//! - `value` stores the native XML node view
//! - `writer` renders XML output
//! - `error` defines XML-specific diagnostics

pub mod error;
mod lexer;
pub mod value;

pub mod parser;
pub mod writer;

pub use error::XmlErrorKind;
pub use parser::{
	benchmark_exit, benchmark_validate_xml_file, parse_xml, validate_xml, validate_xml_file,
};
pub use writer::{append_xml_file, write_xml_file};
pub use value::XmlValue;
