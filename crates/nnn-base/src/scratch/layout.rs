//! Where a carve sits inside a region, decided before any memory is touched.
//!
//! An [`ArenaLayout`] divides one region into the sub-regions the backends need:
//! a stack for carves that are reclaimed in order, a heap for carves that live
//! until the end, and a fixed slot size. Describing this up front is what lets a
//! caller ask whether a whole plan fits before carving any of it.
//!
//! # The types
//!
//! - [`Region`] — an offset and a size, with [`Region::end`] returning `None`
//!   rather than a wrapped bound.
//! - [`StackRegion`] and [`HeapRegion`] — a [`Region`] tagged with the lifetime
//!   it will have, so a layout error is visible in the type a caller holds.
//! - [`Slot`] — one fixed-size slot within the layout.
//! - [`ArenaLayout`] — the division itself.
//!
//! # Validation
//!
//! [`ArenaLayout::new`] rejects a non power of two alignment, a zero slot size,
//! and a slot size that is not a multiple of the alignment, because a slot
//! straddling two boundaries cannot be indexed consistently. Every total is
//! computed with checked arithmetic and surfaces
//! [`CarveError::Overflow`] instead of a plausible-looking number.

use crate::scratch::align::align_up;
use crate::scratch::carve::{
    Carve,
    CarveError,
    CarveKind,
    CarveRegion,
    RegionCarve,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub offset: usize,
    pub size: usize,
}

impl Region {
    pub const fn new(
        offset: usize,
        size: usize,
    ) -> Self {
        Self { offset, size }
    }

    pub const fn end(&self) -> Option<usize> {
        self.offset.checked_add(self.size)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct StackRegion {
    pub region: Region,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct HeapRegion {
    pub region: Region,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub offset: usize,
    pub size: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ArenaLayout {
    pub stack: StackRegion,
    pub heap: HeapRegion,
    pub slot_size: usize,
    pub total: usize,
}

impl ArenaLayout {
    pub const fn new(
        stack_size: usize,
        heap_size: usize,
        slot_size: usize,
        alignment: usize,
    ) -> Result<Self, CarveError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CarveError::InvalidAlignment);
        }

        if slot_size == 0 || !slot_size.is_multiple_of(alignment) {
            return Err(CarveError::InvalidAlignment);
        }

        let heap_offset = match align_up(stack_size, alignment) {
            Some(value) => value,
            None => return Err(CarveError::Overflow),
        };

        let total = match heap_offset.checked_add(heap_size) {
            Some(value) => value,
            None => return Err(CarveError::Overflow),
        };

        Ok(Self {
            stack: StackRegion {
                region: Region::new(0, stack_size),
            },
            heap: HeapRegion {
                region: Region::new(heap_offset, heap_size),
            },
            slot_size,
            total,
        })
    }

    pub const fn total_bytes(&self) -> usize {
        self.total
    }

    pub fn plan_stack(
        &self,
        required: usize,
        alignment: usize,
    ) -> Result<Carve<'_>, CarveError> {
        if required > self.stack.region.size {
            return Err(CarveError::OutOfMemory);
        }

        let carve = Carve::for_bytes(
            required,
            alignment,
            CarveKind::Region(RegionCarve::Stack {
                size: required,
            }),
        )?;

        Ok(carve.with_region(CarveRegion::Stack))
    }

    pub fn plan_slot(
        &self,
        index: usize,
        alignment: usize,
    ) -> Result<Carve<'_>, CarveError> {
        let offset = index
            .checked_mul(self.slot_size)
            .ok_or(CarveError::Overflow)?;

        let slot = self.slot(offset, self.slot_size, alignment)?;

        let mut carve = Carve::for_bytes(
            slot.size,
            alignment,
            CarveKind::Region(RegionCarve::Slot {
                slot_size: slot.size,
                alignment,
            }),
        )?;

        carve.offset = slot.offset;

        Ok(carve.with_region(CarveRegion::Slot { index }))
    }

    pub fn plan_heap(
        &self,
        required: usize,
        alignment: usize,
    ) -> Result<Carve<'_>, CarveError> {
        if required > self.heap.region.size {
            return Err(CarveError::OutOfMemory);
        }

        let carve = Carve::for_bytes(
            required,
            alignment,
            CarveKind::Region(RegionCarve::Heap {
                size: required,
            }),
        )?;

        Ok(carve.with_region(CarveRegion::Heap))
    }

    pub fn slot(
        &self,
        offset: usize,
        size: usize,
        alignment: usize,
    ) -> Result<Slot, CarveError> {
        let aligned = align_up(offset, alignment)
            .ok_or(CarveError::InvalidAlignment)?;

        let end = aligned
            .checked_add(size)
            .ok_or(CarveError::Overflow)?;

        if end > self.total {
            return Err(CarveError::OutOfMemory);
        }

        Ok(Slot {
            offset: aligned,
            size,
        })
    }
}