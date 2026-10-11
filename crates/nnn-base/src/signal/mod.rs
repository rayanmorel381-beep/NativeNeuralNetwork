//! The unit that flows through an LMLP graph.
//!
//! A [`Signal`] carries the payload, the logical time it was emitted, its
//! confidence, its origin and — the key field — the [`Mode`] that decides how
//! an edge will transport it. The node never reads the mode; the edge does.
//!
//! The payload is a [`Buf`], the typed view `scratch` hands out: the crate has
//! no allocator, so a signal owns a carved region rather than a `Box`.

pub(crate) mod meta;
pub(crate) mod mode;
pub(crate) mod node_id;
pub(crate) mod tick;

pub use meta::Meta;
pub use mode::Mode;
pub use node_id::NodeId;
pub use tick::Tick;

use crate::math::float::Float;
use crate::propagation::PropagationError;
use crate::scratch::typed::Buf;

/// One unit of flow: a payload carved from `scratch`, its logical time, its
/// confidence, its origin and the mode an edge reads to transport it.
///
/// The fields are private: the only ways in are [`Signal::new`] and
/// [`Signal::emitted`], both of which enforce the invariant, and the only
/// mutation is [`Signal::set_propagation_mode`], which cannot break it.
pub struct Signal<'a, T: Float> {
    payload: Buf<'a, T>,
    timestamp: Tick,
    confidence: f32,
    origin: NodeId,
    propagation_mode: Mode,
    metadata: Meta,
}

impl<'a, T: Float> Signal<'a, T> {
    /// A signal over `payload`, validating the payload and `confidence`.
    ///
    /// This is the single trusted entry point: every other constructor funnels
    /// through it, so no invalid signal can be built.
    pub fn new(
        payload: Buf<'a, T>,
        timestamp: Tick,
        confidence: f32,
        origin: NodeId,
        propagation_mode: Mode,
        metadata: Meta,
    ) -> Result<Self, PropagationError> {
        let signal = Signal {
            payload,
            timestamp,
            confidence,
            origin,
            propagation_mode,
            metadata,
        };
        signal.validate()?;
        Ok(signal)
    }

    /// A full-confidence signal over `payload`, with neutral metadata.
    pub fn emitted(
        payload: Buf<'a, T>,
        timestamp: Tick,
        origin: NodeId,
        propagation_mode: Mode,
    ) -> Result<Self, PropagationError> {
        Self::new(
            payload,
            timestamp,
            1.0,
            origin,
            propagation_mode,
            Meta::NEUTRAL,
        )
    }

    /// The signal's invariant: a non-empty, finite payload and a finite
    /// confidence within `[0, 1]`.
    pub fn validate(&self) -> Result<(), PropagationError> {
        if self.payload.is_empty() {
            return Err(PropagationError::EmptyPayload);
        }
        if self.payload.as_slice().iter().any(|value| !value.is_finite()) {
            return Err(PropagationError::NonFinitePayload);
        }
        if !self.confidence.is_finite() || !(0.0..=1.0).contains(&self.confidence) {
            return Err(PropagationError::InvalidConfidence);
        }
        Ok(())
    }

    /// The width of the payload, in elements.
    pub fn width(&self) -> usize {
        self.payload.len()
    }

    /// The payload, as a slice.
    pub fn data(&self) -> &[T] {
        self.payload.as_slice()
    }

    /// The logical time the signal was emitted.
    pub fn timestamp(&self) -> Tick {
        self.timestamp
    }

    /// The signal's confidence.
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// The node the signal originated from.
    pub fn origin(&self) -> NodeId {
        self.origin
    }

    /// The mode an edge reads to transport the signal.
    pub fn propagation_mode(&self) -> Mode {
        self.propagation_mode
    }

    /// The routing metadata the signal carries.
    pub fn metadata(&self) -> Meta {
        self.metadata
    }

    /// Rewrites the transport mode, leaving the validated payload and
    /// confidence untouched.
    pub fn set_propagation_mode(&mut self, mode: Mode) {
        self.propagation_mode = mode;
    }
}
