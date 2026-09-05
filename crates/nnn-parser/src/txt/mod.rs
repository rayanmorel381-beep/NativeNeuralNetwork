pub mod error;
pub mod lexer;
mod scalar;
pub mod parser;
pub mod value;

pub use error::{TxtError, TxtErrorKind, TxtErrorPosition};
pub use parser::{
    benchmark_exit, benchmark_validate_txt_file, parse_txt, parse_txt_with_limits, validate_txt,
    validate_txt_file, TxtLimits, TxtParser, DEFAULT_TXT_LIMITS,
};
pub use value::TxtValue;
