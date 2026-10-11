//! Nodes: the pure transformers that turn an inbox of signals into one signal.

pub(crate) mod traits;

pub use traits::{Node, NodeError};
