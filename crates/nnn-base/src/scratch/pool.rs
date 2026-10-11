//! A pool of fixed-size classes over a single linear arena.
//!
//! A [`Pool`] is a [`Scratch`] with a size class per allocation: each class
//! rounds a requested length up to a common multiple, so every block carved from
//! the same class has the same size and blocks stay interchangeable. Carving from
//! two different classes is still strictly stacked, because the underlying
//! cursor is shared — the classes are a rounding policy, not a free list.
//!
//! # Class selection
//!
//! [`Pool::carve`] picks the first class in table order that can hold the
//! request, and reports [`CarveError::EmptyClasses`] when even the largest class
//! is too small. The order of the class table is therefore load-bearing: a table
//! is meant to be ascending, so first-fit and smallest-fit coincide, but an
//! unordered table still works and resolves by position. The reported
//! [`CarveKind`] records the class it landed in, which is what makes a pool
//! carve distinguishable from a plain scratch carve after the fact.
//!
//! # No per-class accounting
//!
//! The pool does not track how many blocks of each class are live. Class choice
//! is derived from the request on the way in, and unwinding is strictly LIFO
//! through the shared cursor, so there is no occupancy state to keep coherent —
//! and therefore no release notification to route. A future class that allowed
//! out-of-order release would be the thing that would require such a
//! notification, and it does not exist here.

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
pub struct PoolMark {
    pub mark: ScratchMark,
    pub class: usize,
}

pub struct Pool<'a> {
    arena: Scratch<'a>,
    classes: &'a [usize],
    default_alignment: usize,
    last_class: Option<usize>,
}

impl<'a> Pool<'a> {
    pub fn new(
        storage: &'a mut [u8],
        classes: &'a [usize],
    ) -> Self {
        Self {
            arena: Scratch::new(storage),
            classes,
            default_alignment: 1,
            last_class: None,
        }
    }

    pub fn with_alignment(
        storage: &'a mut [u8],
        classes: &'a [usize],
        alignment: usize,
    ) -> Self {
        let mut pool = Self::new(storage, classes);

        pool.default_alignment = if alignment == 0 || !alignment.is_power_of_two() {
            1
        } else {
            alignment
        };

        pool
    }

    pub fn capacity(&self) -> usize {
        self.arena.capacity()
    }

    pub fn used(&self) -> usize {
        self.arena.used()
    }

    pub fn remaining(&self) -> usize {
        self.arena.remaining()
    }

    pub fn alignment(&self) -> usize {
        self.default_alignment
    }

    pub fn classes(&self) -> &'a [usize] {
        self.classes
    }

    pub fn capacity_class(
        &self,
        required: usize,
    ) -> Option<usize> {
        self.classes
            .iter()
            .position(|&capacity| capacity != 0 && capacity >= required)
    }

    pub fn class_capacity(
        &self,
        class: usize,
    ) -> Option<usize> {
        self.classes.get(class).copied()
    }

    pub fn mark(&self) -> Option<PoolMark> {
        self.last_class.map(|class| PoolMark {
            mark: self.arena.mark(),
            class,
        })
    }

    pub fn plan<T>(
        &self,
        len: usize,
    ) -> Result<Carve<'a>, CarveError> {
        let bytes = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or(CarveError::Overflow)?;

        self.plan_bytes(bytes, core::mem::align_of::<T>())
    }

    pub fn plan_bytes(
        &self,
        required: usize,
        alignment: usize,
    ) -> Result<Carve<'a>, CarveError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CarveError::InvalidAlignment);
        }

        let class = self
            .capacity_class(required)
            .ok_or(CarveError::EmptyClasses)?;

        let capacity = self.classes[class];

        let carve = Carve::for_bytes(
            capacity,
            alignment,
            CarveKind::From(FromCarve::Bytes {
                size: capacity,
                alignment,
            }),
        )?;

        Ok(carve.with_region(CarveRegion::Pool { class }))
    }

    pub fn carve_bytes(
        &mut self,
        required: usize,
        alignment: usize,
    ) -> Result<Carve<'a>, CarveError> {
        let mut carve = self.plan_bytes(required, alignment)?;

        self.arena.carve_apply(&mut carve)?;

        self.last_class = Some(match carve.region {
            CarveRegion::Pool { class } => class,
            _ => 0,
        });

        Ok(carve)
    }

    pub fn carve_class(
        &mut self,
        class: usize,
    ) -> Result<Carve<'a>, CarveError> {
        let capacity = self
            .class_capacity(class)
            .filter(|&capacity| capacity != 0)
            .ok_or(CarveError::EmptyClasses)?;

        let alignment = self.default_alignment;

        let mut carve = Carve::for_bytes(
            capacity,
            alignment,
            CarveKind::From(FromCarve::Bytes {
                size: capacity,
                alignment,
            }),
        )?
        .with_region(CarveRegion::Pool { class });

        self.arena.carve_apply(&mut carve)?;

        self.last_class = Some(class);

        Ok(carve)
    }

    #[inline]
    pub fn carve<T>(
        &mut self,
        len: usize,
    ) -> Result<&'a mut [T], CarveError> {
        let mut carve = self.plan::<T>(len)?;

        self.arena.carve_apply(&mut carve)?;

        self.last_class = Some(match carve.region {
            CarveRegion::Pool { class } => class,
            _ => 0,
        });

        let ptr = carve.as_mut_ptr();

        Ok(unsafe {
            core::slice::from_raw_parts_mut(ptr as *mut T, len)
        })
    }

    /// # Safety
    ///
    /// See [`Scratch::release`](crate::scratch::arena::Scratch::release).
    pub unsafe fn release(
        &mut self,
        mark: PoolMark,
    ) -> Result<(), CarveError> {
        self.last_class = Some(mark.class);

        self.arena.release(mark.mark)
    }

    /// # Safety
    ///
    /// Every carve handed out by this pool must have been released first,
    /// otherwise the same slot is handed out twice as `&mut`. Releasing in the
    /// order carves were acquired satisfies this, and so does
    /// [`Pool::release`] applied to the most recent mark.
    pub unsafe fn reset(&mut self) {
        self.arena.reset();
        self.last_class = None;
    }
}