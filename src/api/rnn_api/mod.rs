pub(crate) mod core_api;
pub mod build;
pub mod read;
pub mod run;
pub mod train;

pub use build::{
    build_f32,
    build_f64,
};
pub use read::read_rnn;
pub use run::run;
pub use train::train;
