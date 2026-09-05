//! INI support for Omnifile.
//!
//! This module implements a small, predictable parser for classic section-based
//! INI configuration files. Its goal is to make validation and conversion easy,
//! not to support every vendor-specific extension found in the wild.
//!
//! ## Responsibilities
//!
//! - scan significant lines while ignoring comments and blanks
//! - parse `[section]` headers and `key=value` entries
//! - expose the native `IniValue` representation
//! - serialize normalized INI output
//! - report precise line-oriented diagnostics when the structure is invalid
//!
//! ## Submodules
//!
//! - `parser` is the high-level entry point
//! - `lexer` performs line-based scanning
//! - `value` stores parsed sections and entries
//! - `writer` renders data back into INI text
//! - `error` defines format-specific failures

pub mod error;
mod lexer;
pub mod value;

pub mod parser;
pub mod writer;

pub use error::IniErrorKind;
pub use parser::{
	benchmark_exit, benchmark_validate_ini_file, parse_ini, validate_ini, validate_ini_file,
};
pub use writer::{append_ini_file, write_ini_file};
pub use value::IniValue;
