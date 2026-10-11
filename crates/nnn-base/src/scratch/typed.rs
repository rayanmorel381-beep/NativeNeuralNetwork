//! The only route from scratch bytes to typed values.
//!
//! [`Buf`] wraps a carved region and exposes it as `&[T]` or `&mut [T]`. Every
//! typed operation in the workspace goes through it: `math/` depends on this
//! module and on nothing else, so it never handles a raw pointer and can never
//! step outside the region its owner was given.
//!
//! # Construction
//!
//! - [`Buf::from_carve`] adopts a [`Carve`], checking the alignment of the
//!   region against `T` and that its length divides evenly.
//! - [`Buf::from_bytes`] does the same from a byte slice.
//! - [`Buf::from_slice`] adopts a slice directly, for memory that did not come
//!   from a scratch arena.
//!
//! # Why a pointer is held internally
//!
//! A `Buf` records where the region starts and how many elements it holds rather
//! than keeping a slice alive, so a single owner can be tracked across arenas.
//! The two accessors that rebuild a slice from that record are the only place in
//! this module where the safety of the caller's invariants is required; both are
//! reachable solely through methods that take `&self` or `&mut self`, never
//! through the raw pointer itself.

use core::marker::PhantomData;
use core::slice;

use crate::scratch::carve::{Carve, CarveError};

pub struct Buf<'a, T> {
    ptr: *mut T,
    len: usize,
    marker: PhantomData<&'a mut T>,
}

impl<'a, T> Buf<'a, T> {
    pub fn from_carve(mut carve: Carve<'a>) -> Result<Self, CarveError> {
        let size = core::mem::size_of::<T>();
        let alignment = core::mem::align_of::<T>();

        let address = carve.as_mut_ptr() as usize;

        if alignment != 0 && !address.is_multiple_of(alignment) {
            return Err(CarveError::Misaligned);
        }

        if carve.is_empty() {
            return Err(CarveError::SizeMismatch);
        }

        if size != 0 && !carve.len().is_multiple_of(size) {
            return Err(CarveError::SizeMismatch);
        }

        let len = if size == 0 { 0 } else { carve.len() / size };

        Ok(Self {
            ptr: address as *mut T,
            len,
            marker: PhantomData,
        })
    }

    pub fn from_slice(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            marker: PhantomData,
        }
    }

    pub fn from_bytes(
        bytes: &'a mut [u8],
    ) -> Result<Self, CarveError> {
        let size = core::mem::size_of::<T>();
        let alignment = core::mem::align_of::<T>();
        let address = bytes.as_mut_ptr() as usize;

        if alignment != 0 && !address.is_multiple_of(alignment) {
            return Err(CarveError::Misaligned);
        }

        if size == 0 {
            return Ok(Self {
                ptr: address as *mut T,
                len: 0,
                marker: PhantomData,
            });
        }

        if !bytes.len().is_multiple_of(size) {
            return Err(CarveError::SizeMismatch);
        }

        Ok(Self {
            ptr: bytes.as_mut_ptr() as *mut T,
            len: bytes.len() / size,
            marker: PhantomData,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            slice::from_raw_parts(self.ptr, self.len)
        }
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            slice::from_raw_parts_mut(self.ptr, self.len)
        }
    }

    #[inline]
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> Option<(&mut [T], &mut [T])> {
        if mid > self.len {
            return None;
        }

        Some(self.as_mut_slice().split_at_mut(mid))
    }
}