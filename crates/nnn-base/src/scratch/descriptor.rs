//! The fixed set of regions one layer of a network needs.
//!
//! A [`ScratchDesc`] names the eight buffers a computation reads and writes:
//! [`state`](ScratchDesc::state) and [`input`](ScratchDesc::input) on the way
//! in, [`output`](ScratchDesc::output) on the way out, [`weights`](ScratchDesc::weights)
//! and [`bias`](ScratchDesc::bias) for the parameters, [`gradient`](ScratchDesc::gradient)
//! for the backward pass, and [`memory`](ScratchDesc::memory) and
//! [`aux`](ScratchDesc::aux) as scratch the layer may use freely.
//!
//! # Why a struct
//!
//! Naming the regions makes the contract between a layer and its caller
//! explicit. A signature taking eight slices is easy to get wrong at the call
//! site; a signature taking a `ScratchDesc` says which is which, and because the
//! fields are distinct names rather than a positional list, swapping two of them
//! is a type error rather than a silent bug.
//!
//! Every field is a distinct `&'a mut [T]`, so the borrow checker guarantees
//! the eight regions do not overlap: a caller cannot pass the same memory as
//! both input and output, and cannot hold two views of one region at once. That
//! is the ownership rule of this crate, applied to the layer boundary.
//!
//! Regions that a given layer does not need can be passed as empty slices; see
//! [`ScratchDesc::zero`] for clearing only the ones in use.

use crate::scratch::arena::Scratch;
use crate::scratch::carve::CarveError;

pub struct ScratchDesc<'a, T> {
    pub state: &'a mut [T],
    pub input: &'a mut [T],
    pub output: &'a mut [T],
    pub weights: &'a mut [T],
    pub bias: &'a mut [T],
    pub gradient: &'a mut [T],
    pub memory: &'a mut [T],
    pub aux: &'a mut [T],
}

pub struct DescSizes {
    pub state: usize,
    pub input: usize,
    pub output: usize,
    pub weights: usize,
    pub bias: usize,
    pub gradient: usize,
    pub memory: usize,
    pub aux: usize,
}

impl DescSizes {
    pub const fn zeroed() -> Self {
        Self {
            state: 0,
            input: 0,
            output: 0,
            weights: 0,
            bias: 0,
            gradient: 0,
            memory: 0,
            aux: 0,
        }
    }

    pub fn total(&self) -> Result<usize, CarveError> {
        [
            self.state,
            self.input,
            self.output,
            self.weights,
            self.bias,
            self.gradient,
            self.memory,
            self.aux,
        ]
        .iter()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .ok_or(CarveError::Overflow)
    }

    pub fn total_for<T>(&self) -> Result<usize, CarveError> {
        self.total()?
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(CarveError::Overflow)
    }
}

impl<'a, T> ScratchDesc<'a, T> {
    /// Reassembles the eight regions, given in `state, input, output, weights,
    /// bias, gradient, memory, aux` order.
    pub fn new(
        [
            state,
            input,
            output,
            weights,
            bias,
            gradient,
            memory,
            aux,
        ]: [&'a mut [T]; 8],
    ) -> Self {
        Self {
            state,
            input,
            output,
            weights,
            bias,
            gradient,
            memory,
            aux,
        }
    }

    pub fn from_sizes(
        storage: &'a mut [T],
        sizes: &DescSizes,
    ) -> Result<Self, CarveError> {
        if storage.len() < sizes.total()? {
            return Err(CarveError::OutOfMemory);
        }

        let (state, rest) = storage.split_at_mut(sizes.state);
        let (input, rest) = rest.split_at_mut(sizes.input);
        let (output, rest) = rest.split_at_mut(sizes.output);
        let (weights, rest) = rest.split_at_mut(sizes.weights);
        let (bias, rest) = rest.split_at_mut(sizes.bias);
        let (gradient, rest) = rest.split_at_mut(sizes.gradient);
        let (memory, rest) = rest.split_at_mut(sizes.memory);
        let (aux, _) = rest.split_at_mut(sizes.aux);

        Ok(Self {
            state,
            input,
            output,
            weights,
            bias,
            gradient,
            memory,
            aux,
        })
    }

    pub fn carve(
        scratch: &mut Scratch<'a>,
        sizes: &DescSizes,
    ) -> Result<Self, CarveError> {
        let address = scratch.available().as_ptr() as usize;

        if !address.is_multiple_of(core::mem::align_of::<T>()) {
            return Err(CarveError::Misaligned);
        }

        if sizes.total_for::<T>()? > scratch.remaining() {
            return Err(CarveError::OutOfMemory);
        }

        Ok(Self {
            state: scratch.carve::<T>(sizes.state)?,
            input: scratch.carve::<T>(sizes.input)?,
            output: scratch.carve::<T>(sizes.output)?,
            weights: scratch.carve::<T>(sizes.weights)?,
            bias: scratch.carve::<T>(sizes.bias)?,
            gradient: scratch.carve::<T>(sizes.gradient)?,
            memory: scratch.carve::<T>(sizes.memory)?,
            aux: scratch.carve::<T>(sizes.aux)?,
        })
    }

    pub fn zero(&mut self, zero: T)
    where
        T: Copy,
    {
        self.state.fill(zero);
        self.input.fill(zero);
        self.output.fill(zero);
        self.weights.fill(zero);
        self.bias.fill(zero);
        self.gradient.fill(zero);
        self.memory.fill(zero);
        self.aux.fill(zero);
    }
}