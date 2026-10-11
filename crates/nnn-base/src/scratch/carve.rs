//! The carve machine: planning, carving, extending, reclaiming.
//!
//! A [`Carve`] is a request for memory that has not necessarily been handed out
//! yet. It starts [`Planned`](CarveState::Planned), becomes
//! [`Carved`](CarveState::Carved) when a backend applies it, and ends
//! [`Released`](CarveState::Released) once the bytes are given back.
//!
//! # What a carve describes
//!
//! [`CarveKind`] carries the intent, and the four families are independent of
//! any backend:
//!
//! - `Split` — take a region at an offset: [`At`](SplitCarve::At),
//!   [`Head`](SplitCarve::Head), [`Tail`](SplitCarve::Tail) or
//!   [`Middle`](SplitCarve::Middle).
//! - `From` — take a whole region of a given size or type.
//! - `Reuse` — reclaim what was already carved: [`Marked`](ReuseCarve::Marked)
//!   rewinds to a byte offset, [`Last`](ReuseCarve::Last) undoes the last carve,
//!   [`Reset`](ReuseCarve::Reset) releases everything.
//! - `Region` — bind to a backing region, stack, heap or slot.
//!
//! # Splitting
//!
//! [`check_split`] validates a request against a region without touching it and
//! returns a [`Span`]. [`carve_span`] then applies that span, and
//! [`carve_split`] combines the two. Splitting never moves bytes: it hands out
//! three disjoint slices, the part before the carve, the carve itself, and the
//! part after it.
//!
//! # Reclaiming
//!
//! A carve can only be reclaimed through its kind. [`Carve::reuse`] builds the
//! request, and the backend decides how the cursor moves — see
//! [`Scratch::reuse`](crate::scratch::arena::Scratch::reuse).
//!
//! # Requirements
//!
//! [`RequiredBytes`] lives here too, because a size and its alignment are not a
//! separate concern from the request that carries them: a [`Carve`] holds one,
//! and there is no way to name a carve's footprint without it. Keeping the two
//! in one module also means `CarveError` and `RequiredBytes` can see each
//! other without either of them importing the other.

use crate::scratch::align::align_up;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CarveState {
    Planned,
    Carved,
    Released,
}

pub struct Carve<'a> {
    pub kind: CarveKind,
    pub required: RequiredBytes,
    pub shape: Shape,
    pub region: CarveRegion,
    pub offset: usize,
    pub data: &'a mut [u8],
    pub state: CarveState,
}

impl<'a> Carve<'a> {
    pub fn new(
        kind: CarveKind,
        required: RequiredBytes,
    ) -> Self {
        Self {
            kind,
            required,
            shape: Shape::zero(),
            region: CarveRegion::Linear,
            offset: 0,
            data: &mut [],
            state: CarveState::Planned,
        }
    }

    pub fn uncarved(
        kind: CarveKind,
        required: RequiredBytes,
    ) -> Self {
        Self::new(kind, required)
    }

    pub fn for_type<T>(
        count: usize,
        kind: CarveKind,
    ) -> Result<Self, CarveError> {
        let required = RequiredBytes::for_type::<T>(count)?;

        Ok(Self::new(kind, required))
    }

    pub fn for_shape<T>(
        shape: Shape,
        kind: CarveKind,
    ) -> Result<Self, CarveError> {
        let count = shape.count()?;

        let required = RequiredBytes::for_type::<T>(count)?;

        Ok(Self::new(kind, required).with_shape(shape))
    }

    pub fn for_bytes(
        bytes: usize,
        alignment: usize,
        kind: CarveKind,
    ) -> Result<Self, CarveError> {
        let required = RequiredBytes::new(bytes, alignment)?;

        Ok(Self::new(kind, required))
    }

    pub fn for_slice<T>(
        slice: &[T],
        kind: CarveKind,
    ) -> Result<Self, CarveError> {
        Self::for_type::<T>(slice.len(), kind)
    }

    pub fn reuse(
        reuse: ReuseCarve,
        shape: Shape,
    ) -> Result<Self, CarveError> {
        let count = shape.count()?;

        if count == 0 {
            return Err(CarveError::SizeMismatch);
        }

        // Every rewind request is sized the same way, and none of them is sized
        // from the position it names. A rewind is a *position*, and all it needs
        // in storage is room to record the offset it was carved at.
        //
        // `Marked(n)` used to set its requirement to `n`, which made carving the
        // request itself the thing that pushed the cursor to or past `n`. The
        // bound inside `Scratch::reuse` — "a request cannot name a position the
        // cursor never reached" — was then unreachable by construction, and the
        // check was dead code that only looked like a validation. With the
        // requirement decoupled, a request for a far-off position carves fine and
        // is refused where it should be, at reuse time.
        let required = RequiredBytes::new(count, core::mem::size_of::<usize>().max(1))?;

        Ok(Self::new(CarveKind::Reuse(reuse), required).with_shape(shape))
    }

    #[inline]
    pub fn reuse_kind(&self) -> Option<ReuseCarve> {
        match self.kind {
            CarveKind::Reuse(reuse) => Some(reuse),
            _ => None,
        }
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shape = shape;
        self
    }

    pub fn with_region(mut self, region: CarveRegion) -> Self {
        self.region = region;
        self
    }

    #[inline]
    pub fn is_planned(&self) -> bool {
        self.state == CarveState::Planned
    }

    #[inline]
    pub fn is_carved(&self) -> bool {
        self.state == CarveState::Carved
    }

    #[inline]
    pub fn is_released(&self) -> bool {
        self.state == CarveState::Released
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.data
    }

    #[inline]
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.data
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    #[inline]
    pub fn shape(&self) -> Shape {
        self.shape
    }

    #[inline]
    pub fn required(&self) -> usize {
        self.required.bytes()
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.required.alignment()
    }

    pub fn release(&mut self) {
        self.data = &mut [];
        self.offset = 0;
        self.shape = Shape::zero();
        self.state = CarveState::Released;
    }

    pub fn elements<T>(&self) -> Result<usize, CarveError> {
        let size = core::mem::size_of::<T>();

        if self.alignment() < core::mem::align_of::<T>() {
            return Err(CarveError::InvalidAlignment);
        }

        if size == 0 {
            return self.shape.count();
        }

        // The count has to be derived from the same quantity `required` was
        // checked against. A split carve hands out `len` bytes, which can be
        // smaller than `required`, so dividing `required` while measuring `len`
        // is what made the two disagree.
        let bytes = if self.is_carved() {
            self.len()
        } else {
            self.required()
        };

        if bytes % size != 0 {
            return Err(CarveError::SizeMismatch);
        }

        Ok(bytes / size)
    }

    pub fn is_valid_layout<T>(&self) -> bool {
        if self.shape.is_empty() {
            return self.elements::<T>().is_ok();
        }

        match (self.shape.count(), self.elements::<T>()) {
            (Ok(shape_count), Ok(carved_count)) => shape_count == carved_count,
            _ => false,
        }
    }

    pub fn validate(&self) -> Result<(), CarveError> {
        let alignment = self.alignment();

        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CarveError::InvalidAlignment);
        }

        match &self.kind {
            CarveKind::Split(SplitCarve::At(offset))
            | CarveKind::Split(SplitCarve::Head(offset))
            | CarveKind::Split(SplitCarve::Tail(offset)) => {
                if *offset > self.required() {
                    return Err(CarveError::SizeMismatch);
                }
            }

            CarveKind::Split(SplitCarve::Middle { at, len }) => {
                let end = at.checked_add(*len).ok_or(CarveError::Overflow)?;

                if end > self.required() {
                    return Err(CarveError::SizeMismatch);
                }
            }

            CarveKind::From(FromCarve::Bytes { alignment, .. })
            | CarveKind::From(FromCarve::Typed { alignment, .. })
            | CarveKind::Region(RegionCarve::Slot { alignment, .. }) => {
                if *alignment == 0 || !alignment.is_power_of_two() {
                    return Err(CarveError::InvalidAlignment);
                }
            }

            CarveKind::Reuse(reuse) => match reuse {
                ReuseCarve::Marked(bytes) => {
                    if *bytes == 0 {
                        return Err(CarveError::SizeMismatch);
                    }
                }
                ReuseCarve::Last | ReuseCarve::Reset => {}
            },

            _ => {}
        }

        Ok(())
    }

    pub fn extend(
        self,
        other: Carve<'a>,
    ) -> Result<Carve<'a>, CarveError> {
        if !self.is_planned() || !other.is_planned() {
            return Err(CarveError::SizeMismatch);
        }

        let required = self.required.extend(other.required)?;

        let shape = self.shape.extend(other.shape)?;

        let kind = CarveKind::From(FromCarve::Bytes {
            size: required.bytes(),
            alignment: required.alignment(),
        });

        Ok(Carve::new(kind, required)
            .with_shape(shape)
            .with_region(self.region))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shape {
    pub dims: [usize; 5],
}

impl Shape {
    pub const fn new(
        dims: [usize; 5],
    ) -> Self {
        Self { dims }
    }

    pub const fn zero() -> Self {
        Self { dims: [0; 5] }
    }

    pub const fn count(&self) -> Result<usize, CarveError> {
        let mut total = 1usize;

        let mut index = 0;

        while index < 5 {
            total = match total.checked_mul(self.dims[index]) {
                Some(product) => product,
                None => return Err(CarveError::Overflow),
            };

            index += 1;
        }

        Ok(total)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.dims.contains(&0)
    }

    pub fn extend(
        self,
        other: Shape,
    ) -> Result<Shape, CarveError> {
        let mut dims = [0usize; 5];

        for (index, dim) in dims.iter_mut().enumerate() {
            *dim = match self.dims[index].checked_mul(other.dims[index]) {
                Some(product) => product,
                None => return Err(CarveError::Overflow),
            };
        }

        Ok(Shape { dims })
    }

    pub fn for_matrix(
        rows: usize,
        cols: usize,
    ) -> Self {
        Self::new([1, 1, 1, rows, cols])
    }

    pub fn for_vector(
        len: usize,
    ) -> Self {
        Self::new([1, 1, 1, 1, len])
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitCarve {
    At(usize),
    Head(usize),
    Tail(usize),
    Middle {
        at: usize,
        len: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FromCarve {
    Bytes {
        size: usize,
        alignment: usize,
    },
    Typed {
        count: usize,
        alignment: usize,
    },
    Slice {
        len: usize,
    },
    Slot {
        slot_size: usize,
        count: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ReuseCarve {
    Marked(usize),
    Last,
    Reset,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RegionCarve {
    Stack {
        size: usize,
    },
    Heap {
        size: usize,
    },
    Slot {
        slot_size: usize,
        alignment: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CarveRegion {
    Linear,
    Stack,
    Heap,
    Slot {
        index: usize,
    },
    Pool {
        class: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CarveKind {
    Split(SplitCarve),
    From(FromCarve),
    Reuse(ReuseCarve),
    Region(RegionCarve),
}

/// A byte count and the boundary it must start on.
///
/// This is what a [`Carve`] carries before any memory is touched, and it answers
/// the two questions every request has: how many bytes, and at what alignment.
///
/// Keeping the two fields in one type makes it impossible to state a size
/// without an alignment. Every constructor validates — an alignment must be a
/// power of two, and a multiplication that would wrap returns
/// [`CarveError::Overflow`] rather than a plausible-looking number — so an
/// under-aligned or overflowing request is rejected when it is described, before
/// a backend commits to it.
///
/// [`RequiredBytes::for_type`] derives both from `T`, so a typed request cannot
/// disagree with its element type. [`RequiredBytes::extend`] joins two
/// requirements into one that holds both, rounding up to the stricter alignment —
/// the same rounding [`plan_bytes`] applies across a whole plan.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RequiredBytes {
    bytes: usize,
    alignment: usize,
}

impl RequiredBytes {
    pub const fn zero() -> Self {
        Self {
            bytes: 0,
            alignment: 1,
        }
    }

    pub const fn new(
        bytes: usize,
        alignment: usize,
    ) -> Result<Self, CarveError> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(CarveError::InvalidAlignment);
        }

        Ok(Self {
            bytes,
            alignment,
        })
    }

    #[inline]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    #[inline]
    pub const fn alignment(&self) -> usize {
        self.alignment
    }

    pub const fn for_type<T>(
        count: usize,
    ) -> Result<Self, CarveError> {
        let bytes = match core::mem::size_of::<T>().checked_mul(count) {
            Some(value) => value,
            None => return Err(CarveError::Overflow),
        };

        Self::new(
            bytes,
            core::mem::align_of::<T>(),
        )
    }

    pub const fn extend(
        self,
        other: Self,
    ) -> Result<Self, CarveError> {
        let offset = match align_up(self.bytes, other.alignment) {
            Some(offset) => offset,
            None => return Err(CarveError::InvalidAlignment),
        };

        let bytes = match offset.checked_add(other.bytes) {
            Some(bytes) => bytes,
            None => return Err(CarveError::Overflow),
        };

        let alignment = if self.alignment > other.alignment {
            self.alignment
        } else {
            other.alignment
        };

        Ok(Self {
            bytes,
            alignment,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CarveError {
    OutOfMemory,
    InvalidAlignment,
    Misaligned,
    SizeMismatch,
    Overflow,
    EmptyClasses,
    InvalidNumber,
}

/// Parses a byte offset written as text into a `usize`.
///
/// The four bases a Rust integer literal accepts are understood: a `0x`/`0X`
/// prefix selects hexadecimal, `0o`/`0O` octal, `0b`/`0B` binary, and anything
/// else is decimal. A single `_` may separate two digits.
///
/// An empty string, a prefix with no digits, an unknown digit, a misplaced `_`,
/// or a value that does not fit a `usize` all return a [`CarveError`] — the
/// last one as [`CarveError::Overflow`], the rest as
/// [`CarveError::InvalidNumber`].
pub fn parse_offset(text: &str) -> Result<usize, CarveError> {
    let bytes = text.as_bytes();

    if bytes.is_empty() {
        return Err(CarveError::InvalidNumber);
    }

    let (radix, digits) = if bytes.len() >= 2 && bytes[0] == b'0' {
        match bytes[1] {
            b'x' | b'X' => (16u32, &bytes[2..]),
            b'o' | b'O' => (8u32, &bytes[2..]),
            b'b' | b'B' => (2u32, &bytes[2..]),
            _ => (10u32, bytes),
        }
    } else {
        (10u32, bytes)
    };

    let mut value: usize = 0;
    let mut seen = false;
    let mut separator = false;

    for &byte in digits {
        if byte == b'_' {
            if !seen || separator {
                return Err(CarveError::InvalidNumber);
            }

            separator = true;
            continue;
        }

        let digit = (byte as char)
            .to_digit(radix)
            .ok_or(CarveError::InvalidNumber)? as usize;

        value = value
            .checked_mul(radix as usize)
            .and_then(|value| value.checked_add(digit))
            .ok_or(CarveError::Overflow)?;

        seen = true;
        separator = false;
    }

    if !seen || separator {
        return Err(CarveError::InvalidNumber);
    }

    Ok(value)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub prefix: usize,
    pub start: usize,
    pub len: usize,
}

pub struct CarveRest<'a> {
    pub skipped: &'a mut [u8],
    pub after: &'a mut [u8],
}

pub fn check_split(
    req: &Carve<'_>,
    arena_len: usize,
    arena_addr: usize,
) -> Result<Span, CarveError> {
    req.validate()?;

    let origin = arena_addr;
    let aligned = align_up(origin, req.alignment()).ok_or(CarveError::Overflow)?;

    let prefix = aligned - origin;

    if prefix > arena_len {
        return Err(CarveError::OutOfMemory);
    }

    let body = arena_len - prefix;

    let (start, len) = match &req.kind {
        CarveKind::Split(SplitCarve::At(offset))
        | CarveKind::Split(SplitCarve::Head(offset)) => (0usize, *offset),

        CarveKind::Split(SplitCarve::Tail(offset)) => (
            body.checked_sub(*offset).ok_or(CarveError::SizeMismatch)?,
            *offset,
        ),

        CarveKind::Split(SplitCarve::Middle { at, len }) => (*at, *len),

        _ => (0usize, req.required()),
    };

    let end = start.checked_add(len).ok_or(CarveError::Overflow)?;

    if end > body {
        return Err(CarveError::OutOfMemory);
    }

    Ok(Span { prefix, start, len })
}

pub fn carve_span<'a>(
    arena: &'a mut [u8],
    span: Span,
    req: &mut Carve<'a>,
) -> CarveRest<'a> {
    let (_, body) = arena.split_at_mut(span.prefix);

    let (skipped, tail) = body.split_at_mut(span.start);

    let (carved, after) = tail.split_at_mut(span.len);

    req.offset = span.prefix + span.start;
    req.data = carved;
    req.state = CarveState::Carved;

    CarveRest { skipped, after }
}

pub fn carve_split<'a>(
    arena: &'a mut [u8],
    req: &mut Carve<'a>,
) -> Result<CarveRest<'a>, CarveError> {
    let span = check_split(req, arena.len(), arena.as_ptr() as usize)?;

    Ok(carve_span(arena, span, req))
}

pub fn carve_from<'a, T>(
    arena: &'a mut [u8],
    req: &mut Carve<'a>,
) -> Result<usize, CarveError> {
    let expected_alignment = core::mem::align_of::<T>();
    let expected_size = core::mem::size_of::<T>();

    if req.alignment() < expected_alignment {
        return Err(CarveError::InvalidAlignment);
    }

    if expected_size != 0 && !req.required().is_multiple_of(expected_size) {
        return Err(CarveError::SizeMismatch);
    }

    carve_split(arena, req)?;

    if expected_alignment != 0 && !(req.as_ptr() as usize).is_multiple_of(expected_alignment) {
        return Err(CarveError::Misaligned);
    }

    req.elements::<T>()
}

pub const fn carve_required(
    req: &Carve<'_>,
) -> RequiredBytes {
    req.required
}

pub const fn required_of(
    req: &Carve<'_>,
) -> usize {
    req.required.bytes()
}

pub const fn alignment_of(
    req: &Carve<'_>,
) -> usize {
    req.required.alignment()
}

pub const fn plan_bytes<const N: usize>(
    plan: &[Carve<'_>; N],
) -> Option<usize> {
    let mut total = 0usize;

    let mut index = 0;

    while index < N {
        let alignment = plan[index].required.alignment();

        let aligned = match align_up(total, alignment) {
            Some(aligned) => aligned,
            None => return None,
        };

        let current = plan[index].required.bytes();

        total = match aligned.checked_add(current) {
            Some(total) => total,
            None => return None,
        };

        index += 1;
    }

    Some(total)
}
