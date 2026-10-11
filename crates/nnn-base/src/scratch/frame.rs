//! A frame: a carve plus the ability to give it back.
//!
//! A [`Frame`] pairs a [`Carve`] with the region it was carved from, so the two
//! cannot be separated. Once a frame has been carved, the bytes are committed
//! and the region is exhausted up to the carve; releasing the frame returns
//! those bytes to the region and nothing else.
//!
//! # Release is explicit
//!
//! There is deliberately no [`Drop`](core::ops::Drop) impl. An implicit drop
//! would release the bytes at a point the caller cannot see — a dropped
//! temporary would rewind the region while a sibling carve is still being built,
//! which is exactly the aliasing this module exists to prevent. Drop-based
//! cleanup here would be a masking pattern, not a convenience: the caller has to
//! decide when the frame's lifetime ends.
//!
//! Because a frame owns bytes, releasing it is [`unsafe`](Frame::release): the
//! caller must guarantee that no slice, reference or carve still points into the
//! frame's region.

use crate::scratch::arena::{
    Scratch,
    ScratchMark,
};
use crate::scratch::carve::{
    Carve,
    CarveError,
    CarveKind,
    CarveRegion,
    FromCarve,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FrameState {
    Alive,
    Released,
}

pub struct Frame<'a, 's> {
    scratch: &'s mut Scratch<'a>,
    mark: ScratchMark,
    state: FrameState,
}

impl<'a, 's> Frame<'a, 's> {
    pub fn new(
        scratch: &'s mut Scratch<'a>,
    ) -> Self {
        let mark = scratch.mark();

        Self {
            scratch,
            mark,
            state: FrameState::Alive,
        }
    }

    pub fn state(&self) -> FrameState {
        self.state
    }

    pub fn is_alive(&self) -> bool {
        self.state == FrameState::Alive
    }

    pub fn mark(&self) -> ScratchMark {
        self.mark
    }

    pub fn plan(
        &self,
        required: usize,
        alignment: usize,
    ) -> Result<Carve<'a>, CarveError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CarveError::InvalidAlignment);
        }

        let carve = Carve::for_bytes(
            required,
            alignment,
            CarveKind::From(FromCarve::Bytes {
                size: required,
                alignment,
            }),
        )?;

        Ok(carve.with_region(CarveRegion::Linear))
    }

    pub fn plan_of<T>(
        &self,
        len: usize,
    ) -> Result<Carve<'a>, CarveError> {
        let required = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(CarveError::Overflow)?;

        self.plan(required, core::mem::align_of::<T>())
    }

    pub fn carve(
        &mut self,
        mut req: Carve<'a>,
    ) -> Result<Carve<'a>, CarveError> {
        if !self.is_alive() {
            return Err(CarveError::SizeMismatch);
        }

        self.scratch.carve_apply(&mut req)?;

        Ok(req)
    }

    pub fn carve_bytes(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<&'a mut [u8], CarveError> {
        let carve = self.plan(size, alignment)?;

        let carve = self.carve(carve)?;

        Ok(carve.data)
    }

    #[inline]
    pub fn carve_of<T>(
        &mut self,
        len: usize,
    ) -> Result<&'a mut [T], CarveError> {
        let carve = self.plan_of::<T>(len)?;

        let mut carve = self.carve(carve)?;

        let ptr = carve.as_mut_ptr();

        Ok(unsafe {
            core::slice::from_raw_parts_mut(ptr as *mut T, len)
        })
    }

    /// # Safety
    ///
    /// See [`Scratch::release`](crate::scratch::arena::Scratch::release).
    pub unsafe fn release(&mut self) -> Result<(), CarveError> {
        if !self.is_alive() {
            return Ok(());
        }

        self.scratch.release(self.mark)?;

        self.state = FrameState::Released;

        Ok(())
    }
}