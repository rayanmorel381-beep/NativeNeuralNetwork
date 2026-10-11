//! Linear cursor backend: forwards when carving, rewinds when reclaiming.
//!
//! A [`Scratch`] owns one region supplied by the caller and hands out carves
//! from it in order. It records where the next carve will start and keeps a
//! pointer to the region so a rewind can restore the full extent.
//!
//! # Forwarding
//!
//! [`Scratch::carve_apply`] validates the request with
//! [`check_split`](crate::scratch::carve::check_split), applies it, and advances
//! the cursor past the bytes handed out. It only ever moves forwards.
//!
//! # Reclaiming
//!
//! [`Scratch::reuse`] is the counterpart, and the only path that moves the
//! cursor backwards. It accepts a carve built with
//! [`Carve::reuse`](crate::scratch::carve::Carve::reuse) and interprets its kind
//! as an instruction: rewind to a recorded offset, undo the last carve, or
//! release everything. Rewinding is what makes [`ReuseCarve`] more than a
//! declaration.
//!
//! # Safety
//!
//! Reclaiming returns bytes that may still be described by a live carve. Every
//! rewind path is therefore `unsafe` and its contract is that the caller has
//! ended the lifetime of the carves it is giving back. That contract cannot be
//! expressed in the type system — a carve is a cursor position, not an
//! ownership token — so it is stated in English at each of the four places
//! where a slice is rebuilt from the base pointer.

use core::slice;

use crate::scratch::carve::{
    carve_span,
    check_split,
    Carve,
    CarveError,
    CarveKind,
    CarveState,
    FromCarve,
    ReuseCarve,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ScratchMark {
    cursor: usize,
}

pub struct Scratch<'a> {
    storage: &'a mut [u8],
    base: *mut u8,
    total: usize,
    cursor: usize,
}

impl<'a> Scratch<'a> {
    pub fn new(storage: &'a mut [u8]) -> Self {
        Self {
            base: storage.as_mut_ptr(),
            total: storage.len(),
            storage,
            cursor: 0,
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.total
    }

    #[inline]
    pub fn used(&self) -> usize {
        self.cursor
    }

    #[inline]
    pub fn remaining(&self) -> usize {
        self.storage.len()
    }

    pub fn mark(&self) -> ScratchMark {
        ScratchMark {
            cursor: self.cursor,
        }
    }

    pub fn carve_apply(
        &mut self,
        req: &mut Carve<'a>,
    ) -> Result<(), CarveError> {
        if req.state != CarveState::Planned {
            return Err(CarveError::SizeMismatch);
        }

        let (len, addr) = {
            let s = &*self.storage;
            (s.len(), s.as_ptr() as usize)
        };

        let span = check_split(req, len, addr)?;

        let storage = core::mem::take(&mut self.storage);
        let rest = carve_span(storage, span, req);

        req.offset += self.cursor;

        self.cursor = req.offset + req.len();
        self.storage = rest.after;

        Ok(())
    }

    pub fn carve_bytes(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<&'a mut [u8], CarveError> {
        let mut carve = Carve::for_bytes(
            size,
            alignment,
            CarveKind::From(FromCarve::Bytes {
                size,
                alignment,
            }),
        )?;

        self.carve_apply(&mut carve)?;

        Ok(carve.data)
    }

    pub fn carve_align(
        &mut self,
        size: usize,
        alignment: usize,
    ) -> Result<&'a mut [u8], CarveError> {
        self.carve_bytes(size, alignment)
    }

    #[inline]
    pub fn carve<T>(
        &mut self,
        len: usize,
    ) -> Result<&'a mut [T], CarveError> {
        let size = core::mem::size_of::<T>()
            .checked_mul(len)
            .ok_or(CarveError::Overflow)?;

        let alignment = core::mem::align_of::<T>();

        let mut carve = Carve::for_bytes(
            size,
            alignment,
            CarveKind::From(FromCarve::Typed {
                count: len,
                alignment,
            }),
        )?;

        self.carve_apply(&mut carve)?;

        let ptr = carve.as_mut_ptr();

        Ok(unsafe {
            slice::from_raw_parts_mut(ptr as *mut T, len)
        })
    }

    /// Rewinds the scratch to the state described by `req` and returns the bytes
    /// that become available again.
    ///
    /// This is the reclamation path declared by `CarveKind::Reuse`: the caller
    /// hands back a `Carve` built with [`Carve::reuse`] and receives the region it
    /// now owns.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that no `Carve` carved at or after the new
    /// cursor is still live, and that nothing derived from one — a slice, a
    /// [`typed::Buf`](crate::scratch::typed::Buf) — outlives the rewind.
    ///
    /// The borrow on the *returned* slice is not the reason this is unsafe. That
    /// slice does borrow `self`, so it cannot overlap the next carve. The
    /// unsoundness is on the other side: a `Carve` handed out earlier is bound to
    /// `'a`, the storage, not to the `&mut self` this call takes. It keeps
    /// pointing into bytes the cursor now reclaims, and the caller would then
    /// hold two `&mut` over them. Only the caller knows whether that is the
    /// case, because a `Carve` carries no runtime liveness flag.
    ///
    /// This is the same contract as [`Scratch::release`] and
    /// [`Scratch::reset`], which are unsafe for the same reason: every rewind
    /// path can invalidate a `Carve` that outlived the call.
    pub unsafe fn reuse(
        &mut self,
        req: &mut Carve<'_>,
    ) -> Result<&mut [u8], CarveError> {
        let reuse = req.reuse_kind().ok_or(CarveError::SizeMismatch)?;

        req.validate()?;

        if !req.is_carved() {
            return Err(CarveError::SizeMismatch);
        }

        let reclaimed = match reuse {
            ReuseCarve::Marked(bytes) => {
                if bytes == 0 || bytes > self.cursor {
                    return Err(CarveError::SizeMismatch);
                }

                self.cursor = bytes;

                unsafe { self.rebuild() }
            }

            ReuseCarve::Last => {
                if req.offset > self.cursor {
                    return Err(CarveError::SizeMismatch);
                }

                self.cursor = req.offset;

                unsafe { self.rebuild() }
            }

            ReuseCarve::Reset => {
                self.cursor = 0;

                unsafe { self.rebuild() }
            }
        };

        req.data = &mut [];
        req.state = CarveState::Released;

        Ok(reclaimed)
    }

    /// Reapplies the ownership rule after the cursor moved.
    ///
    /// This is the single place in this module where a slice is rebuilt from the
    /// base pointer, so the `Scratch` invariant — one region has at most one
    /// owner — is stated once. Every rewind path goes through it: the three
    /// [`ReuseCarve`] instructions, [`Scratch::release`] and [`Scratch::reset`].
    ///
    /// # Safety
    ///
    /// The caller must guarantee that no `Carve` carved after the new cursor is
    /// still live. A `Carve` that was carved before the new cursor becomes part
    /// of the reclaimed region and must not be used either.
    unsafe fn rebuild(&mut self) -> &mut [u8] {
        let start = self.base.add(self.cursor);

        self.storage = slice::from_raw_parts_mut(start, self.total - self.cursor);

        self.storage
    }

    #[inline]
    pub fn available(&self) -> &[u8] {
        self.storage
    }

    pub fn available_mut(&mut self) -> &mut [u8] {
        self.storage
    }

    /// # Safety
    ///
    /// Every `Carve` handed out by this scratch must be released or dropped
    /// before the cursor is rewound, otherwise the same bytes are given out
    /// twice as `&mut`. The invariant this call relies on is documented on
    /// [`Scratch::rebuild`].
    pub unsafe fn release(
        &mut self,
        mark: ScratchMark,
    ) -> Result<(), CarveError> {
        if mark.cursor > self.cursor {
            return Err(CarveError::SizeMismatch);
        }

        self.cursor = mark.cursor;

        self.rebuild();

        Ok(())
    }

    /// # Safety
    ///
    /// Every carve handed out by this scratch must have been released first,
    /// otherwise the same bytes are handed out twice as `&mut`. Releasing in the
    /// order carves were acquired satisfies this, and so does
    /// [`Scratch::release`] applied to the most recent mark. The same requirement
    /// is expressed with a `Result` and against a recorded offset by
    /// [`Scratch::reuse`] with [`ReuseCarve::Reset`]; it is `unsafe` for the same
    /// reason this is, not a safe alternative to it. The invariant this call
    /// relies on is documented on [`Scratch::rebuild`].
    pub unsafe fn reset(&mut self) {
        self.cursor = 0;

        self.rebuild();
    }
}