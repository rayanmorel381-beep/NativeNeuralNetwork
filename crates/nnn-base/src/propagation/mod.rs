//! Pure validation and reinterpretation of signals.

pub(crate) mod transform;

pub use transform::{
    lmlp_to_signals, reinterpret_lmlp_signals, signal_to_lmlp, signals_to_lmlp,
    validate_lmlp_signal, LmlpSignals, PropagationError,
};
