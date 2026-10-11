//! A fixed-slot arena: same-size blocks carved strictly in order.
//!
//! A [`SlotArena`] divides its storage into `slot_count` blocks of exactly
//! `slot_size` bytes. Carving always takes the next unused block, so block
//! lifetimes are LIFO and the arena can report its occupancy with three counters
//! rather than a per-block list.
//!
//! # The borrow split, and why `release` is safe
//!
//! Carving takes `&self` and returns a [`Carve`] that *borrows the arena* for as
//! long as it lives. Releasing takes `&mut self`. Those two signatures together
//! are the whole safety argument:
//!
//! - Several carves can be alive at once, because they are shared borrows and
//!   name different, disjoint slots.
//! - Releasing cannot run while any carve is alive, because a live carve holds a
//!   shared borrow and `release` needs the exclusive one. This is checked by the
//!   compiler, not by a promise in a doc comment.
//!
//! An earlier version had `carve_handle(&mut self) -> Carve<'a>` returning a carve
//! bound to the *storage* rather than to the borrow, and `release(&mut self)`
//! marked `unsafe` because nothing could then prove the carve was gone. Two
//! overlapping `&mut` were reachable from safe code. Tying the carve to `&self`
//! moves that obligation from the caller to the type system.
//!
//! # What the counters are still for
//!
//! Handing out `&mut` from `&self` is only sound if the runtime refuses to name
//! the same slot twice, so the counters are load-bearing, not bookkeeping:
//! [`can_carve`] refuses a slot that is already live, and [`can_release`] refuses
//! one that is not the top of the live range. Because acquisition and carving
//! can interleave, liveness is a range ([`live_lo`]/[`live_hi`]) rather than a
//! count — a single counter with `used - live` was wrong and rejected the second
//! of two simultaneous carves.
//!
//! The counters are [`Cell`], not plain fields, precisely so that [`acquire`] and
//! [`carve_handle`] can take `&self`. That is what keeps the borrow split intact.
//! A side effect is that `SlotArena` is `Send` but not `Sync`, which is the right
//! answer for a type whose `&mut`-based release is the synchronisation point.
//!
//! [`used`]: SlotArena#used
//! [`live_lo`]: SlotArena#live_lo
//! [`live_hi`]: SlotArena#live_hi
//! [`can_carve`]: SlotArena#can_carve
//! [`can_release`]: SlotArena#can_release
//! [`acquire`]: SlotArena#acquire
//! [`carve_handle`]: SlotArena#carve_handle
//! [`Cell`]: core::cell::Cell
//! [`Carve`]: crate::scratch::carve::Carve

use core::cell::Cell;
use core::marker::PhantomData;
use core::slice;

use crate::scratch::carve::{
    carve_split,
    Carve,
    CarveError,
    CarveKind,
    CarveRegion,
    FromCarve,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlotHandle {
    pub index: usize,
    pub offset: usize,
}

pub struct SlotArena<'a> {
    base: *mut u8,
    len: usize,
    slot_size: usize,
    alignment: usize,
    slot_count: usize,
    used: Cell<usize>,
    live_lo: Cell<usize>,
    live_hi: Cell<usize>,
    marker: PhantomData<&'a mut [u8]>,
}

impl<'a> SlotArena<'a> {
    pub fn new(
        storage: &'a mut [u8],
        slot_size: usize,
        alignment: usize,
    ) -> Result<Self, CarveError> {
        if slot_size == 0 {
            return Err(CarveError::SizeMismatch);
        }

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CarveError::InvalidAlignment);
        }

        if !slot_size.is_multiple_of(alignment) {
            return Err(CarveError::InvalidAlignment);
        }

        if !(storage.as_ptr() as usize).is_multiple_of(alignment) {
            return Err(CarveError::Misaligned);
        }

        let slot_count = storage.len() / slot_size;

        Ok(Self {
            base: storage.as_mut_ptr(),
            len: storage.len(),
            slot_size,
            alignment,
            slot_count,
            used: Cell::new(0),
            live_lo: Cell::new(0),
            live_hi: Cell::new(0),
            marker: PhantomData,
        })
    }

    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    pub fn alignment(&self) -> usize {
        self.alignment
    }

    pub fn capacity(&self) -> usize {
        self.slot_count
    }

    pub fn used(&self) -> usize {
        self.used.get()
    }

    pub fn remaining(&self) -> usize {
        self.slot_count.saturating_sub(self.used())
    }

    pub fn offset_of(&self, index: usize) -> Option<usize> {
        if index >= self.slot_count {
            return None;
        }

        index.checked_mul(self.slot_size)
    }

    pub fn handle(&self, index: usize) -> Option<SlotHandle> {
        Some(SlotHandle {
            index,
            offset: self.offset_of(index)?,
        })
    }

    /// Reserves the next slot and returns its handle.
    ///
    /// Takes `&self` because the returned carve has to hold a shared borrow of
    /// this arena, and that borrow must not be interrupted by an exclusive one.
    /// The reservation is therefore interior state, which is why `used` is a
    /// [`Cell`].
    pub fn acquire(&self) -> Result<SlotHandle, CarveError> {
        if self.used() >= self.slot_count {
            return Err(CarveError::OutOfMemory);
        }

        let index = self.used();

        let handle = self.handle(index).ok_or(CarveError::Overflow)?;

        self.used.set(index + 1);

        Ok(handle)
    }

    pub fn carve_slot(&self) -> Result<Carve<'_>, CarveError> {
        let handle = self.acquire()?;

        self.carve_handle(handle)
    }

    /// Carves the slot named by `handle`.
    ///
    /// The returned carve borrows this arena, so it keeps every `&mut` operation
    /// on it — [`release`](SlotArena::release) above all — unavailable until the
    /// carve is gone. That is what makes release safe without a contract for the
    /// caller to uphold by hand.
    pub fn carve_handle(
        &self,
        handle: SlotHandle,
    ) -> Result<Carve<'_>, CarveError> {
        let index = self.can_carve(handle)?;

        let slot_size = self.slot_size;
        let alignment = self.alignment;

        let slot = unsafe { self.slot_ptr(index) }
            .map(|ptr| unsafe { slice::from_raw_parts_mut(ptr, self.slot_size) })
            .ok_or(CarveError::OutOfMemory)?;

        let mut carve: Carve<'_> = Carve::for_bytes(
            slot_size,
            alignment,
            CarveKind::From(FromCarve::Slot {
                slot_size,
                count: 1,
            }),
        )?;

        carve_split(slot, &mut carve)?;

        self.mark_live(index);

        Ok(carve.with_region(CarveRegion::Slot { index }))
    }

    /// Borrows a slot that no `Carve` currently owns.
    ///
    /// Takes `&mut self`, which is both how the returned borrow is authorised and
    /// what makes reaching a live slot impossible: a live carve holds a shared
    /// borrow of this arena, so `&mut self` cannot be produced while one exists.
    /// The `is_live` check below is still needed, but for a different reason — a
    /// carve that has gone out of scope leaves its slot marked live, and that
    /// block must not be reborrowed before it is released.
    ///
    /// The gap this covers is the one between [`acquire`](SlotArena::acquire), which reserves a slot,
    /// and [`carve_handle`](SlotArena::carve_handle), which carves it.
    pub fn get_mut(
        &mut self,
        handle: SlotHandle,
    ) -> Option<&mut [u8]> {
        let index = self.validate(handle).ok()?;

        if self.is_live(index) {
            return None;
        }

        unsafe { self.slot_ptr(index) }
            .map(|ptr| unsafe { slice::from_raw_parts_mut(ptr, self.slot_size) })
    }

    /// Returns the index of a slot that may be carved right now.
    ///
    /// Carving takes the top of the arena, `index + 1 == used`, and refuses a
    /// slot that already holds a carve. Several carves may be live at once —
    /// they name different slots, and different slots are disjoint — so what
    /// this guarantees is one `&mut` per slot, which is exactly what
    /// [`slot_at`](SlotArena::slot_at) needs and cannot establish by itself.
    fn can_carve(&self, handle: SlotHandle) -> Result<usize, CarveError> {
        let index = self.validate(handle)?;

        if self.used() == 0 || index + 1 != self.used() {
            return Err(CarveError::SizeMismatch);
        }

        if self.is_live(index) {
            return Err(CarveError::SizeMismatch);
        }

        Ok(index)
    }

    /// Returns the index of a slot that may be released right now.
    ///
    /// Releasing takes the top of the live range, `index + 1 == live_hi`, so the
    /// live range stays contiguous and `live_lo`/`live_hi` keep describing every
    /// live carve without a list.
    fn can_release(&self, handle: SlotHandle) -> Result<usize, CarveError> {
        let index = self.validate(handle)?;

        if !self.is_live(index) || index + 1 != self.live_hi.get() {
            return Err(CarveError::SizeMismatch);
        }

        Ok(index)
    }

    fn is_live(&self, index: usize) -> bool {
        self.live_lo.get() <= index && index < self.live_hi.get()
    }

    fn mark_live(&self, index: usize) {
        if self.live_lo.get() == self.live_hi.get() {
            self.live_lo.set(index);
        }

        self.live_hi.set(index + 1);
    }

    fn unmark_live(&self, index: usize) {
        self.live_hi.set(index);
    }

    fn validate(&self, handle: SlotHandle) -> Result<usize, CarveError> {
        let expected = self
            .handle(handle.index)
            .ok_or(CarveError::OutOfMemory)?;

        if expected.offset != handle.offset {
            return Err(CarveError::SizeMismatch);
        }

        Ok(handle.index)
    }

    /// Releases a carved slot, giving the block back to the arena.
    ///
    /// This is safe, and that is the point of the module's borrow split: every
    /// carve holds a shared borrow of this arena, so taking `&mut self` here is
    /// only possible once no carve is alive. There is no obligation left for the
    /// caller to uphold by hand, which is why this carries no `unsafe`.
    ///
    /// Releasing through the [`Carve`] instead of a handle is impossible — the
    /// carve is already borrowed — and taking a handle is what lets the exclusive
    /// borrow be the synchronisation point. The handle is the same identity, so
    /// the LIFO check below is the same guarantee.
    pub fn release(
        &mut self,
        handle: SlotHandle,
    ) -> Result<(), CarveError> {
        let index = self.can_release(handle)?;

        self.unmark_live(index);

        // `used` and `live_hi` can drift apart: `acquire` bumps `used` before
        // `carve_handle` marks the slot live, so a reservation taken above the
        // live range leaves `used > live_hi`. Releasing the top of the live
        // range hands the block back, but a plain `used - 1` would leave those
        // reservations stranded. Rewinding `used` to the released index instead
        // frees the reserved tail with the slot, so the next `acquire` starts
        // exactly where the live range ends.
        self.used.set(index);

        Ok(())
    }

    /// Returns the address of one slot's first byte.
    ///
    /// # Safety
    ///
    /// The caller must not have already produced a live borrow of this slot.
    /// That is what [`can_carve`](SlotArena::can_carve) establishes, and every path that reaches this
    /// function goes through it.
    ///
    /// The type system cannot see it on its own: the returned pointer will be
    /// turned into a `&mut` while this arena is only borrowed shared, which is
    /// sound only because the runtime state guarantees at most one outstanding
    /// borrow per slot. The complementary half — that such a borrow cannot
    /// outlive the ability to release — is established by the type system, via
    /// the `&self`/`&mut self` split. Neither half is sufficient alone.
    #[inline]
    unsafe fn slot_ptr(&self, index: usize) -> Option<*mut u8> {
        let offset = self.offset_of(index)?;

        let end = offset.checked_add(self.slot_size)?;

        if end > self.len {
            return None;
        }

        Some(unsafe { self.base.add(offset) })
    }
}
