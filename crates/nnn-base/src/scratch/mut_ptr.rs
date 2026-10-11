//! Mutable handles that can be moved between threads.
//!
//! A scratch arena is often owned by one thread while the regions inside it are
//! handed to a worker pool. That requires a handle that is `Send` and `Sync` yet
//! still yields `&mut`, which the borrow checker will not do on its own.
//! [`SyncMutPtr`] and [`SyncSlice`] wrap the target in an [`UnsafeCell`] to take
//! that responsibility explicitly.
//!
//! # The `Sync` contract
//!
//! Both types implement `Send` and `Sync` for any `T: Send`, and neither
//! implementation is a formality — it is a promise that the target is reachable
//! from one thread at a time. The `unsafe fn` accessors say what that promise
//! requires at the call site, and the mutable ones take `&mut self` rather than
//! `&self` so two mutable borrows of the same handle cannot be produced at once.
//! `SyncSlice::as_slice` keeps `&self`, since shared borrows may coexist.
//!
//! This is a narrow primitive, deliberately kept alongside the backends: it is
//! how a carve is published to another thread, not a general purpose cell.

use core::cell::UnsafeCell;
use core::marker::PhantomData;
use core::slice;

pub struct SyncMutPtr<T> {
    ptr: UnsafeCell<*mut T>,
    marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for SyncMutPtr<T> {}
unsafe impl<T: Send> Sync for SyncMutPtr<T> {}

impl<T> SyncMutPtr<T> {
    pub const fn new(ptr: *mut T) -> Self {
        Self {
            ptr: UnsafeCell::new(ptr),
            marker: PhantomData,
        }
    }

    pub fn get(&self) -> *mut T {
        unsafe { *self.ptr.get() }
    }

    pub fn set(&self, ptr: *mut T) {
        unsafe {
            *self.ptr.get() = ptr;
        }
    }

    /// Borrows the target for writing.
    ///
    /// # Safety
    ///
    /// The pointer must be valid for writes and no other reference to the target
    /// may be alive. `&mut self` prevents two mutable borrows at once, but it
    /// cannot know about references held elsewhere.
    #[inline]
    pub unsafe fn as_mut(&mut self) -> &mut T {
        &mut *self.get()
    }
}

pub struct SyncSlice<T> {
    ptr: UnsafeCell<*mut T>,
    len: UnsafeCell<usize>,
    marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for SyncSlice<T> {}
unsafe impl<T: Send> Sync for SyncSlice<T> {}

impl<T> SyncSlice<T> {
    pub const fn new(
        ptr: *mut T,
        len: usize,
    ) -> Self {
        Self {
            ptr: UnsafeCell::new(ptr),
            len: UnsafeCell::new(len),
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        unsafe { *self.len.get() }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self) -> *mut T {
        unsafe { *self.ptr.get() }
    }

    pub fn set(
        &self,
        ptr: *mut T,
        len: usize,
    ) {
        unsafe {
            *self.ptr.get() = ptr;
            *self.len.get() = len;
        }
    }

    /// Borrows the whole target for reading.
    ///
    /// # Safety
    ///
    /// The pointer must be valid for `len` reads and stay valid for the
    /// returned borrow. This is safe to call more than once because shared
    /// borrows may coexist.
    #[inline]
    pub unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(
            self.get(),
            self.len(),
        )
    }

    /// Borrows the whole target for writing.
    ///
    /// # Safety
    ///
    /// The pointer must be valid for `len` writes, and no other reference to
    /// that range may be alive. `&mut self` prevents two mutable borrows of this
    /// handle at once, but it cannot know about references held elsewhere.
    #[inline]
    pub unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(
            self.get(),
            self.len(),
        )
    }
}