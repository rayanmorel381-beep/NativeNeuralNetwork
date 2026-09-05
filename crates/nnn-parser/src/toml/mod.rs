//! TOML support for Omnifile.
//!
//! This module handles TOML documents as structured configuration data. It
//! includes a parser, serializer, native value model, and format-specific error
//! reporting designed to work cleanly with the shared `OmniValue` bridge.
//!
//! ## Main responsibilities
//!
//! - parse tables, arrays, strings, booleans, and numbers
//! - enforce configurable resource limits
//! - report line-aware errors for invalid keys, strings, or nesting
//! - serialize normalized TOML output deterministically
//!
//! ## Reading order for maintainers
//!
//! Start with `parser`, then look at `lexer`, `value`, and finally `writer`.
//! That mirrors the actual data flow used by the rest of the crate.

pub mod error;
mod lexer;
pub mod value;

pub mod parser;
pub mod writer;

pub use error::TomlErrorKind;
pub use parser::{
	benchmark_exit, benchmark_validate_toml_file, parse_toml, validate_toml, validate_toml_file,
};
pub use writer::{append_toml_file, write_toml_file};
pub use value::TomlValue;
