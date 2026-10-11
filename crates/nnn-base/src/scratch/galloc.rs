//! An explicit, narrow frontier where the crate touches the global allocator.
//!
//! Everything else in `scratch/` is `no_alloc`: memory is supplied by the caller
//! and nothing is reserved behind their back. [`GateAlloc`] is the single
//! deliberate exception, and it exists so that exception is visible.
//!
//! # Why a gate
//!
//! A debug build can be configured to fail the moment any layer allocates
//! implicitly, which is what makes a memory pillar testable: if a layer can only
//! obtain memory through an arena it was handed, then a test can account for all
//! of it. Wrapping the allocator in a gate with an
//! [`enabled`](GateAlloc::enabled) flag turns that policy into a runtime switch
//! rather than a build configuration.
//!
//! When the gate is closed, [`allocate`](GateAlloc::allocate) and
//! [`deallocate`](GateAlloc::deallocate) return
//! [`GateAllocError::Disabled`] without touching the underlying allocator. No
//! error is logged or swallowed here — a caller that needs to know simply has to
//! look, which keeps the accounting honest.
//!
//! # The single unsafe operation
//!
//! `deallocate` is `unsafe` because handing a pointer back to an allocator ends
//! the borrow the caller held on it, and the standard library requires that
//! contract to be stated. `allocate` is safe to call because it only ever
//! returns a pointer the caller does not yet own.

use core::alloc::{GlobalAlloc, Layout};

pub enum GateAllocError {
    Disabled,
    InvalidLayout,
    AllocationFailed,
}

pub struct GateAlloc<A> {
    allocator: A,
    enabled: bool,
}

impl<A> GateAlloc<A> {
    pub const fn new(
        allocator: A,
        enabled: bool,
    ) -> Self {
        Self {
            allocator,
            enabled,
        }
    }

    pub const fn disabled(
        allocator: A,
    ) -> Self {
        Self {
            allocator,
            enabled: false,
        }
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn allocate(
        &self,
        size: usize,
        alignment: usize,
    ) -> Result<*mut u8, GateAllocError>
    where
        A: GlobalAlloc,
    {
        if !self.enabled {
            return Err(GateAllocError::Disabled);
        }

        let layout = Layout::from_size_align(
            size,
            alignment,
        )
        .map_err(|_| GateAllocError::InvalidLayout)?;

        let ptr = unsafe {
            self.allocator.alloc(layout)
        };

        if ptr.is_null() {
            return Err(GateAllocError::AllocationFailed);
        }

        Ok(ptr)
    }

    /// # Safety
    ///
    /// `ptr` must have been returned by [`GateAlloc::allocate`] on the same
    /// allocator with the same layout, and must not have been freed since.
    pub unsafe fn deallocate(
        &self,
        ptr: *mut u8,
        size: usize,
        alignment: usize,
    ) -> Result<(), GateAllocError>
    where
        A: GlobalAlloc,
    {
        if !self.enabled {
            return Err(GateAllocError::Disabled);
        }

        let layout = Layout::from_size_align(
            size,
            alignment,
        )
        .map_err(|_| GateAllocError::InvalidLayout)?;

        self.allocator.dealloc(ptr, layout);

        Ok(())
    }
}