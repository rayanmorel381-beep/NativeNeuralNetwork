//! Node memory: the sizes of the optional per-node state regions.
//!
//! The regions live in a carved [`Buf`]; this module only describes how the
//! `scratch` region a node owns is split, it never allocates.

use crate::scratch::typed::Buf;

/// The sizes of the three optional memory regions a node can own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NodeMemory {
    pub recurrent_len: usize,
    pub opt_moments_len: usize,
    pub norm_stats_len: usize,
}

impl NodeMemory {
    /// A node with no memory at all.
    pub const NONE: NodeMemory = NodeMemory {
        recurrent_len: 0,
        opt_moments_len: 0,
        norm_stats_len: 0,
    };

    /// The total number of slots across the three regions.
    pub const fn total(&self) -> usize {
        self.recurrent_len + self.opt_moments_len + self.norm_stats_len
    }

    /// Splits a carved `memory` region into `(recurrent, opt_moments, norm_stats)`.
    ///
    /// Returns `None` when the region is smaller than [`total`](Self::total).
    pub fn split<'m, 'a, T>(
        &self,
        memory: &'m mut Buf<'a, T>,
    ) -> Option<(&'m mut [T], &'m mut [T], &'m mut [T])> {
        if memory.len() < self.total() {
            return None;
        }
        let slice = memory.as_mut_slice();
        let (recurrent, rest) = slice.split_at_mut(self.recurrent_len);
        let (opt_moments, norm_stats) = rest.split_at_mut(self.opt_moments_len);
        Some((recurrent, opt_moments, norm_stats))
    }
}
