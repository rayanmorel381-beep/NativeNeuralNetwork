//! The scratch layer: the memory interface every other layer builds on.
//!
//! [`LmlpScratch`] owns the region handed in by the host and is the one type
//! that turns raw bytes into carves, typed buffers and a [`ScratchDesc`]. Every
//! layer above it (`math`, `activations`, `initializers`, `tensor`) receives
//! this type and nothing lower, so the ownership rule of `scratch` is stated
//! once, here.
//!
//! Beyond the plain cursor, it also composes the other backends of the pillar —
//! [`ArenaLayout`], [`Frame`], [`Pool`], [`SlotArena`], [`LogBuf`] and the
//! `GateAlloc` — so that every reclamation strategy is a real path `base.rs`
//! can reach, not a dormant design.

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;

use crate::scratch::align::{align_down, align_of_size, is_aligned};
use crate::scratch::arena::{Scratch, ScratchMark};
use crate::scratch::carve::{
    alignment_of, carve_from, carve_required, carve_split, plan_bytes as plan_carve_bytes,
    required_of, Carve, CarveError, CarveKind, FromCarve, ReuseCarve, RequiredBytes, Shape,
    SplitCarve,
};
use crate::scratch::descriptor::{DescSizes, ScratchDesc};
use crate::scratch::frame::{Frame, FrameState};
use crate::scratch::galloc::GateAlloc;
use crate::scratch::layout::ArenaLayout;
use crate::scratch::logbuf::LogBuf;
use crate::scratch::mut_ptr::{SyncMutPtr, SyncSlice};
use crate::scratch::pool::Pool;
use crate::scratch::slot::SlotArena;
use crate::scratch::typed::Buf;

/// An allocator that never hands out memory.
///
/// The crate allocates nothing, so the only `GlobalAlloc` it ever composes is
/// one whose allocation always fails: the gate stays closed and every query
/// through it is a pure statement about the gate, not an allocation attempt.
struct NopAllocator;

unsafe impl GlobalAlloc for NopAllocator {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        null_mut()
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

/// The seven figures a [`LmlpScratch::pool_carve`] reports, in order: capacity,
/// used-before, remaining-before, effective alignment, class-table length,
/// class capacity, and the bytes written across the pool, a byte carve and a
/// class carve.
///
/// A named tuple keeps one line per figure in the signatures that carry it,
/// without a public type to leak through `nnn-base`'s facade.
pub type PoolCarveReport = (usize, usize, usize, usize, usize, usize, usize);

pub struct LmlpScratch<'a> {
    arena: Scratch<'a>,
}

impl<'a> LmlpScratch<'a> {
    pub fn new(storage: &'a mut [u8]) -> Self {
        Self::from_arena(Scratch::new(storage))
    }

    pub fn from_arena(arena: Scratch<'a>) -> Self {
        Self { arena }
    }

    pub fn arena(&self) -> &Scratch<'a> {
        &self.arena
    }

    pub fn arena_mut(&mut self) -> &mut Scratch<'a> {
        &mut self.arena
    }

    pub fn capacity(&self) -> usize {
        self.arena().capacity()
    }

    pub fn used(&self) -> usize {
        self.arena().used()
    }

    pub fn remaining(&self) -> usize {
        self.arena().remaining()
    }

    pub fn mark(&self) -> ScratchMark {
        self.arena().mark()
    }

    pub fn carve_bytes(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<&'a mut [u8], CarveError> {
        self.arena_mut().carve_align(size, alignment)
    }

    pub fn carve<T>(&mut self, len: usize) -> Result<&'a mut [T], CarveError> {
        self.arena_mut().carve::<T>(len)
    }

    pub fn carve_apply(&mut self, req: &mut Carve<'a>) -> Result<(), CarveError> {
        self.arena_mut().carve_apply(req)
    }

    /// Carves `len` values of `T` and hands them back as the only typed view
    /// bytes are allowed to take.
    pub fn buf<T>(&mut self, len: usize) -> Result<Buf<'a, T>, CarveError> {
        Buf::from_carve(self.carve_typed::<T>(len)?)
    }

    /// Carves a `Carve` for `len` values of `T` and applies it to the region,
    /// leaving the caller free to adopt it as a typed [`Buf`] or hand it to a
    /// layer above.
    pub fn carve_typed<T>(&mut self, len: usize) -> Result<Carve<'a>, CarveError> {
        let mut carve = Carve::for_type::<T>(
            len,
            CarveKind::From(FromCarve::Typed {
                count: len,
                alignment: align_of_size::<T>(),
            }),
        )?;

        debug_assert!(carve.is_planned());

        let region = self.arena_mut().carve_bytes(
            core::mem::size_of::<T>()
                .checked_mul(len)
                .ok_or(CarveError::Overflow)?,
            align_of_size::<T>(),
        )?;

        let carved = carve_from::<T>(region, &mut carve)?;

        debug_assert_eq!(carved, len);
        debug_assert!(carve.is_valid_layout::<T>());
        debug_assert_eq!(carve.bytes().len(), carved * core::mem::size_of::<T>());
        debug_assert!(carve.shape().is_empty());

        Ok(carve)
    }

    /// Carves the eight named regions one layer needs in one call.
    pub fn desc<T: 'a>(
        &mut self,
        sizes: &DescSizes,
    ) -> Result<ScratchDesc<'a, T>, CarveError> {
        let _footprint = self.footprint::<T>(sizes)?;

        let alignment = align_of_size::<T>();

        let address = self.arena_mut().available_mut().as_ptr() as usize;

        if align_down(address, alignment) != Some(address)
            || !is_aligned(address, alignment)
        {
            return Err(CarveError::Misaligned);
        }

        ScratchDesc::carve(self.arena_mut(), sizes)
    }

    /// The bytes a whole [`DescSizes`] plan would need, aligned for `T`.
    pub fn footprint<T>(&self, sizes: &DescSizes) -> Result<usize, CarveError> {
        let plan: [Carve<'_>; 8] = [
            self.plan_one::<T>(sizes.state)?,
            self.plan_one::<T>(sizes.input)?,
            self.plan_one::<T>(sizes.output)?,
            self.plan_one::<T>(sizes.weights)?,
            self.plan_one::<T>(sizes.bias)?,
            self.plan_one::<T>(sizes.gradient)?,
            self.plan_one::<T>(sizes.memory)?,
            self.plan_one::<T>(sizes.aux)?,
        ];

        let total = plan_carve_bytes(&plan).ok_or(CarveError::Overflow)?;

        let mut regions = plan.into_iter();

        let mut joined = regions.next().ok_or(CarveError::Overflow)?;

        for region in regions {
            joined = joined.extend(region)?;
        }

        debug_assert_eq!(total, joined.required());

        Ok(total)
    }

    /// Splits the region into one carved `T` block and reports its three parts.
    pub fn split_take<T>(&mut self, len: usize) -> Result<(usize, usize, usize), CarveError> {
        let mut req = Carve::for_type::<T>(
            len,
            CarveKind::From(FromCarve::Typed {
                count: len,
                alignment: align_of_size::<T>(),
            }),
        )?;

        let region = self.arena_mut().carve_bytes(
            core::mem::size_of::<T>()
                .checked_mul(len)
                .ok_or(CarveError::Overflow)?,
            align_of_size::<T>(),
        )?;

        let rest = carve_split(region, &mut req)?;

        let skipped = rest.skipped.len();
        let after = rest.after.len();
        let carved = req.len();

        Ok((skipped, after, carved))
    }

    /// Carves `len` values of `T`, reports the payload, and hands the carve
    /// straight back, in its released state.
    pub fn divest_typed<T>(&mut self, len: usize) -> Result<usize, CarveError> {
        let mut carve = self.carve_typed::<T>(len)?;

        let payload = carve.bytes().len();

        carve.release();

        let _released = carve.is_released();

        let emptied = carve.bytes_mut().len();

        Ok(payload + emptied)
    }

    fn plan_one<T>(&self, count: usize) -> Result<Carve<'_>, CarveError> {
        Carve::for_type::<T>(
            count,
            CarveKind::From(FromCarve::Typed {
                count,
                alignment: align_of_size::<T>(),
            }),
        )
    }

    /// The footprint a single `T` carve of `len` elements will own.
    pub fn needed<T>(&self, len: usize) -> Result<usize, CarveError> {
        let carve = Carve::for_type::<T>(
            len,
            CarveKind::From(FromCarve::Typed {
                count: len,
                alignment: align_of_size::<T>(),
            }),
        )?;

        let required = carve_required(&carve);

        debug_assert_eq!(required.alignment(), alignment_of(&carve));

        Ok(required.bytes())
    }

    /// The footprint a matrix or vector shape would own.
    pub fn matrix_footprint<T>(
        &self,
        rows: usize,
        cols: usize,
    ) -> Result<usize, CarveError> {
        self.shaped_footprint::<T>(Shape::for_matrix(rows, cols))
    }

    /// The footprint a single-row vector of `len` elements would own.
    pub fn vector_footprint<T>(&self, len: usize) -> Result<usize, CarveError> {
        self.shaped_footprint::<T>(Shape::for_vector(len))
    }

    fn shaped_footprint<T>(&self, shape: Shape) -> Result<usize, CarveError> {
        if shape.is_empty() {
            return Err(CarveError::SizeMismatch);
        }

        let count = shape.count()?;

        let carve = Carve::for_shape::<T>(
            shape,
            CarveKind::From(FromCarve::Typed {
                count,
                alignment: align_of_size::<T>(),
            }),
        )?;

        Ok(carve_required(&carve).bytes())
    }

    /// The aligned footprint a caller's own `slice` would need.
    pub fn slice_footprint<T>(&self, slice: &[T]) -> Result<usize, CarveError> {
        let carve = Carve::for_slice::<T>(
            slice,
            CarveKind::From(FromCarve::Slice { len: slice.len() }),
        )?;

        Ok(carve_required(&carve).bytes())
    }

    /// Describes four split requests without touching any memory.
    pub fn split_scan(&self, at: usize, tail: usize, middle: usize) -> Result<usize, CarveError> {
        let at_req = Carve::new(
            CarveKind::Split(SplitCarve::At(at)),
            RequiredBytes::zero(),
        );
        let head_req = Carve::new(
            CarveKind::Split(SplitCarve::Head(at)),
            RequiredBytes::zero(),
        );
        let tail_req = Carve::new(
            CarveKind::Split(SplitCarve::Tail(tail)),
            RequiredBytes::zero(),
        );
        let middle_req = Carve::new(
            CarveKind::Split(SplitCarve::Middle { at, len: middle }),
            RequiredBytes::zero(),
        );

        middle_req.validate()?;

        Ok([
            required_of(&at_req),
            required_of(&head_req),
            required_of(&tail_req),
            required_of(&middle_req),
        ]
        .iter()
        .sum())
    }

    /// Plans stack, heap and slot regions the way [`ArenaLayout`] sees them.
    pub fn layout_scan(
        &self,
        stack: usize,
        heap: usize,
        slot_size: usize,
        alignment: usize,
    ) -> Result<(usize, usize, usize, usize), CarveError> {
        let layout = ArenaLayout::new(stack, heap, slot_size, alignment)?;

        let stack_end = layout.stack.region.end().ok_or(CarveError::Overflow)?;
        let heap_end = layout.heap.region.end().ok_or(CarveError::Overflow)?;

        let _stack = layout.plan_stack(1, alignment)?;
        let _heap = layout.plan_heap(1, alignment)?;
        let _slot = layout.plan_slot(1, alignment)?;

        let first = layout.slot(0, slot_size, alignment)?;
        let _first_end = first.offset.checked_add(first.size).ok_or(CarveError::Overflow)?;

        Ok((stack_end, heap_end, layout.slot_size, layout.total_bytes()))
    }

    /// Carves `len` elements through a size-class [`Pool`].
    ///
    /// # Safety
    ///
    /// The region handed to the pool is released before it is returned, so no
    /// byte the pool touched may still be live.
    pub unsafe fn pool_carve<T>(
        &mut self,
        classes: &'a [usize],
        len: usize,
    ) -> Result<PoolCarveReport, CarveError> {
        let alignment = align_of_size::<T>();

        let storage = self.arena_mut().carve_bytes(len * core::mem::size_of::<T>(), alignment)?;

        let mut pool = Pool::with_alignment(storage, classes, alignment);

        let capacity = pool.capacity();
        let used_before = pool.used();
        let remaining_before = pool.remaining();
        let effective_alignment = pool.alignment();
        let class_table = pool.classes().len();

        let data = pool.carve::<T>(len)?;

        let mark = pool.mark().ok_or(CarveError::SizeMismatch)?;
        let class = mark.class;

        let class_capacity = pool.class_capacity(class).ok_or(CarveError::EmptyClasses)?;

        let bytes_written = {
            let carve = pool.carve_bytes(core::mem::size_of::<T>(), alignment)?;

            carve.len()
        };

        let class_written = {
            let carve = pool.carve_class(class)?;

            carve.len()
        };

        let written = data.len();

        pool.release(mark)?;

        pool.reset();

        Ok((
            capacity,
            used_before,
            remaining_before,
            effective_alignment,
            class_table,
            class_capacity,
            written + bytes_written + class_written,
        ))
    }

    /// Carves and returns a `T` block through a fixed-size [`SlotArena`].
    pub fn slot_carve<T>(
        &mut self,
        slot_count: usize,
    ) -> Result<(usize, usize, usize, usize, usize), CarveError> {
        let slot_size = core::mem::size_of::<T>();
        let alignment = align_of_size::<T>();

        let storage = self.arena_mut().carve_bytes(
            slot_size.saturating_mul(slot_count),
            alignment,
        )?;

        let mut arena = SlotArena::new(storage, slot_size, alignment)?;

        let capacity = arena.capacity();
        let slot_bytes = arena.slot_size();
        let alignment_of_slot = arena.alignment();

        // Handles reserve slots in order; carving takes the top of the live
        // range, so the uppermost handle is the one a carve can hold.
        let first = arena.acquire()?;
        let _second = arena.acquire()?;

        let first_len = {
            let carve = arena.carve_handle(first)?;

            carve.data.len()
        };

        let carve_slot = arena.carve_slot()?;
        let second_len = carve_slot.data.len();

        // The `carve_slot` above acquired slot index 2; build its handle for the
        // release below.
        let third = arena.handle(2).ok_or(CarveError::OutOfMemory)?;

        // All carves are dropped, so the arena can be borrowed mutably again.
        let used = arena.used();
        let remaining = arena.remaining();

        let carved_len = first_len + second_len;

        // `release` reuses the top of the live range: `third` is live and is
        // `live_hi - 1`, so `can_release` accepts it and the block returns.
        arena.release(third)?;

        let released_len = {
            let slot = arena
                .get_mut(third)
                .ok_or(CarveError::SizeMismatch)?;

            slot.len()
        };

        Ok((
            capacity,
            slot_bytes,
            alignment_of_slot,
            remaining,
            used + carved_len + released_len,
        ))
    }

    /// Writes `bytes` into a fixed [`LogBuf`] carved out of the region.
    pub fn log_bytes(&mut self, capacity: usize, bytes: &[u8]) -> Result<(usize, usize), CarveError> {
        // Build a `Carve` and adopt the region through `LogBuf::from_carve`, so
        // the constructor that takes a carve owns the alignment of its bytes.
        let mut carve = Carve::uncarved(
            CarveKind::From(FromCarve::Bytes { size: capacity, alignment: 1 }),
            RequiredBytes::new(capacity, 1)?,
        );

        self.carve_apply(&mut carve)?;

        let mut carve_log = LogBuf::from_carve(carve)?;

        let _capacity = carve_log.capacity();
        let _empty = carve_log.is_empty();

        let written = carve_log.write_bytes(bytes);

        let dropped = carve_log.dropped();

        let _len = carve_log.len();

        carve_log.clear();

        // The plain constructor takes the region directly.
        let buffer = self.arena_mut().carve_bytes(capacity, 1)?;

        let mut log = LogBuf::new(buffer);

        let direct_written = log.write_bytes(bytes);

        let direct_dropped = log.dropped();

        if log.as_bytes().is_empty() {
            let _ = log.write_fmt(core::format_args!("{}", direct_written));
        }

        Ok((written + direct_written, dropped + direct_dropped))
    }

    /// The allocation gate is closed: the crate never allocates.
    pub fn gate_closed(&self) -> bool {
        !GateAlloc::disabled(NopAllocator).enabled()
    }

    /// Asks the gate whether an implicit allocation of `size` bytes would slip
    /// through. With the gate closed this is always `None`.
    ///
    /// # Safety
    ///
    /// `size` and `alignment` must describe a valid layout; with the gate
    /// closed no allocation is retained, so no memory is leaked either way.
    pub unsafe fn allocation_test(
        &self,
        size: usize,
        alignment: usize,
    ) -> (Option<usize>, bool) {
        let gate = GateAlloc::new(NopAllocator, false);

        let allocated = match gate.allocate(size, alignment) {
            Ok(ptr) => Some(unsafe {
                let _ = gate.deallocate(ptr, size, alignment);
                size
            }),
            Err(_) => None,
        };

        let deallocation_gated = !gate.enabled();

        (allocated, deallocation_gated)
    }

    /// Publishes `data` as a [`SyncSlice`] and reads its two ends back.
    ///
    /// # Safety
    ///
    /// The caller must not touch `data` while this handle would be shared.
    pub unsafe fn front_back<T: Copy>(&mut self, data: &'a mut [T]) -> (T, T) {
        let mut pointer = SyncMutPtr::new(data.as_mut_ptr());

        pointer.set(data.as_mut_ptr());

        let first_owned = *unsafe { pointer.as_mut() };

        let mut shared = SyncSlice::new(pointer.get(), data.len());

        let count = shared.len();
        let _empty = shared.is_empty();

        shared.set(pointer.get(), count);

        let whole = unsafe { shared.as_slice() };

        let first = whole.first().copied().unwrap_or(first_owned);
        let last = whole.last().copied().unwrap_or(first_owned);

        let _ = unsafe { shared.as_mut_slice() };

        (first, last)
    }

    /// Carves `len` elements inside a [`Frame`] and releases it.
    ///
    /// # Safety
    ///
    /// Nothing carved inside the frame may be live when the frame is released.
    pub unsafe fn frame_carve<T: 'a>(
        &mut self,
        len: usize,
    ) -> Result<(ScratchMark, usize), CarveError> {
        let mut frame = Frame::new(self.arena_mut());

        let state = frame.state();
        let _alive = frame.is_alive();

        if state != FrameState::Alive {
            return Err(CarveError::SizeMismatch);
        }

        let mark = frame.mark();

        let _planned = frame.plan(len, align_of_size::<T>())?;
        let _planned_of = frame.plan_of::<T>(len)?;

        let data = frame.carve_of::<T>(len)?;

        let bytes = frame.carve_bytes(core::mem::size_of::<T>() * len, align_of_size::<T>())?;

        let n = data.len();

        let bits = bytes.len();

        frame.release()?;

        Ok((mark, n + bits))
    }

    /// # Safety
    ///
    /// See [`Scratch::release`](crate::scratch::arena::Scratch::release).
    pub unsafe fn release(&mut self, mark: ScratchMark) -> Result<(), CarveError> {
        self.arena_mut().release(mark)
    }

    /// # Safety
    ///
    /// See [`Scratch::reuse`](crate::scratch::arena::Scratch::reuse).
    pub unsafe fn reuse_marked(&mut self, bytes: usize) -> Result<usize, CarveError> {
        let mut req = Carve::reuse(
            ReuseCarve::Marked(bytes),
            Shape::new([1, 1, 1, 1, bytes]),
        )?;

        // A rewind request must be carved before it can be reused: carving
        // records its position, and `reuse` refuses a request that never was.
        self.arena_mut().carve_apply(&mut req)?;

        Ok(self.arena_mut().reuse(&mut req)?.len())
    }

    /// # Safety
    ///
    /// See [`Scratch::reuse`](crate::scratch::arena::Scratch::reuse).
    pub unsafe fn reuse_last(&mut self) -> Result<usize, CarveError> {
        let mut req = Carve::reuse(ReuseCarve::Last, Shape::new([1, 1, 1, 1, 1]))?;

        self.arena_mut().carve_apply(&mut req)?;

        Ok(self.arena_mut().reuse(&mut req)?.len())
    }

    /// # Safety
    ///
    /// See [`Scratch::reuse`](crate::scratch::arena::Scratch::reuse).
    pub unsafe fn reuse_reset(&mut self) -> Result<usize, CarveError> {
        let mut req = Carve::reuse(ReuseCarve::Reset, Shape::new([1, 1, 1, 1, 1]))?;

        // The request itself must book a slot to record where it was carved.
        // A fully carved region still releases everything, so fall back to the
        // direct path when there is no room to book it.
        if self.arena_mut().remaining() == 0 {
            self.arena_mut().reset();
            return Ok(0);
        }

        self.arena_mut().carve_apply(&mut req)?;

        Ok(self.arena_mut().reuse(&mut req)?.len())
    }

    /// # Safety
    ///
    /// Every carve handed out by this scratch must have been released first.
    pub unsafe fn reset(&mut self) {
        self.arena_mut().reset();
    }
}