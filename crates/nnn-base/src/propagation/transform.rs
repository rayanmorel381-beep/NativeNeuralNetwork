//! Pure validation and reinterpretation of signals.
//!
//! Every payload is a [`Buf`]; the module never allocates and never copies a
//! payload — it validates the carved region in place.

use crate::math::float::Float;
use crate::scratch::typed::Buf;
use crate::signal::{Meta, Mode, NodeId, Signal, Tick};

/// Why a signal could not be validated or reinterpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropagationError {
    EmptyPayload,
    NonFinitePayload,
    InvalidConfidence,
    BufferTooSmall,
}

/// A validated, contiguous run of signals.
#[derive(Clone, Copy)]
pub struct LmlpSignals<'a, T: Float> {
    signals: &'a [Signal<'a, T>],
}

impl<'a, T: Float> LmlpSignals<'a, T> {
    /// The underlying signals.
    pub fn as_signals(&self) -> &'a [Signal<'a, T>] {
        self.signals
    }

    /// How many signals the run holds.
    pub fn len(&self) -> usize {
        self.signals.len()
    }

    /// Whether the run is empty.
    pub fn is_empty(&self) -> bool {
        self.signals.is_empty()
    }
}

/// Wraps a carved `payload` into a signal, validating it first.
pub fn signal_to_lmlp<'a, T: Float>(
    payload: Buf<'a, T>,
    timestamp: Tick,
    origin: NodeId,
    propagation_mode: Mode,
    confidence: f32,
    metadata: Meta,
) -> Result<Signal<'a, T>, PropagationError> {
    Signal::new(
        payload,
        timestamp,
        confidence,
        origin,
        propagation_mode,
        metadata,
    )
}

/// Re-validates an existing signal in place.
pub fn validate_lmlp_signal<T: Float>(signal: &Signal<'_, T>) -> Result<(), PropagationError> {
    signal.validate()
}

/// Validates every signal in `signals` as one contiguous run.
pub fn signals_to_lmlp<'a, T: Float>(
    signals: &'a [Signal<'a, T>],
) -> Result<LmlpSignals<'a, T>, PropagationError> {
    for signal in signals {
        validate_lmlp_signal(signal)?;
    }
    Ok(LmlpSignals { signals })
}

/// The signals held by a validated run.
pub fn lmlp_to_signals<'a, T: Float>(lmlp: &'a LmlpSignals<'a, T>) -> &'a [Signal<'a, T>] {
    lmlp.as_signals()
}

/// Rewrites every signal's mode in place, validating first, and reports the
/// number of signals rewritten.
pub fn reinterpret_lmlp_signals<T: Float>(
    signals: &mut [Signal<'_, T>],
    propagation_mode: Mode,
) -> Result<usize, PropagationError> {
    for signal in signals.iter_mut() {
        validate_lmlp_signal(signal)?;
        signal.set_propagation_mode(propagation_mode);
    }
    Ok(signals.len())
}
