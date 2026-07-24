#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantError {
    Empty,
    ShapeMismatch,
    InvalidScale,
}
