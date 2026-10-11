use crate::math::float::Float;
use crate::signal::{Signal, Tick};

/// Why a node could not produce a signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeError {
    EmptyInbox,
    InvalidSignal,
    OutputTooSmall,
}

/// A pure transformer from an inbox of signals to a single signal.
pub trait Node<T: Float> {
    /// Folds `inbox` into the node's next signal.
    fn update<'a>(&mut self, inbox: &'a [Signal<'a, T>]) -> Result<Signal<'a, T>, NodeError>;

    /// Consumes an incoming gradient; the default is a no-op.
    fn learn(&mut self, _gradient: &[Signal<'_, T>]) -> Result<(), NodeError> {
        Ok(())
    }

    /// Advances any internal time bookkeeping to `_tick`.
    fn synchronize(&mut self, _tick: Tick) {}
}
