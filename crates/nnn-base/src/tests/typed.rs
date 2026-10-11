//! Behaviour tests for typed views over carved bytes.
//!
//! `Buf<T>` is where the `unsafe` of the carve machine actually turns a raw byte
//! block into something with a type. The unsafe is real and it lives in one
//! place: `as_slice` and `as_mut_slice` rebuild a slice from a stored pointer, so
//! if a `Buf` were ever built over a block whose address does not satisfy
//! `align_of::<T>()`, or whose length is not a whole number of `T`, the first
//! load is undefined behaviour. Every constructor therefore has to refuse those
//! two cases, and that refusal is what most of these tests check.
//!
//! # Reaching a `Carve`
//!
//! `Scratch::carve_bytes` and `Scratch::carve` hand back plain slices, not
//! `Carve` values, so `Buf::from_carve` has no shorter route in. The tests build
//! the `Carve` the way the arena itself does — `Carve::for_bytes` to describe,
//! then `Scratch::carve_apply` to commit — which also keeps this honest: if those
//! two steps ever stop composing, the failure shows up here.

use crate::scratch::arena::Scratch;
use crate::scratch::carve::{Carve, CarveError, CarveKind, FromCarve};
use crate::scratch::typed::Buf;

use super::support::Aligned;

/// Describe a carve, apply it, and hand back the `Carve` with its bytes.
///
/// The `Carve<'a>` borrows the arena's storage, not the `&mut Scratch`, so the
/// returned carve stays usable after this function returns.
fn carved<'a>(scratch: &mut Scratch<'a>, bytes: usize, alignment: usize) -> Carve<'a> {
    let kind = CarveKind::From(FromCarve::Bytes { size: bytes, alignment });

    let mut carve = Carve::for_bytes(bytes, alignment, kind).expect("a valid request");

    scratch.carve_apply(&mut carve).expect("a carve that fits in the arena");

    carve
}

#[test]
fn a_carved_block_becomes_a_buf_of_the_right_length() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let buf = Buf::<f32>::from_carve(carved(&mut scratch, 24, 4)).expect("24 is six f32");

    assert_eq!(buf.len(), 6);
    assert!(!buf.is_empty());
    assert_eq!(buf.as_slice().len(), 6);
    assert_eq!(
        buf.as_slice().as_ptr() as usize % 4,
        0,
        "the block was requested 4-aligned, so the elements are addressable"
    );
}

#[test]
fn a_length_that_is_not_a_whole_number_of_elements_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    assert_eq!(
        Buf::<f32>::from_carve(carved(&mut scratch, 18, 4)).err(),
        Some(CarveError::SizeMismatch),
        "18 is four and a half f32, so there is no correct element count. \
         Rounding up to 5 would read two bytes past the end of the block; \
         rounding down to 4 would silently drop the two the caller asked for. \
         Refusing is the only safe answer."
    );
}

#[test]
fn a_block_whose_address_misses_the_alignment_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    // Shift the cursor off the alignment by carving a single byte first. The
    // block that follows starts one byte into an 8-aligned buffer, which is not
    // 4-aligned.
    {
        let _odd = scratch.carve_bytes(1, 1).expect("a one-byte carve fits");
    }

    let mut carve = carved(&mut scratch, 24, 1);

    assert_eq!(
        carve.as_mut_ptr() as usize % 4,
        1,
        "the fixture has to actually be misaligned for this test to mean \
         anything"
    );
    assert_eq!(
        Buf::<f32>::from_carve(carve).err(),
        Some(CarveError::Misaligned),
        "24 divides by 4, so only the address can be the reason — and a &[f32] \
         over a misaligned block faults on the first load"
    );
}

#[test]
fn an_empty_carve_is_refused_while_an_empty_byte_slice_is_not() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    assert_eq!(
        Buf::<f32>::from_carve(carved(&mut scratch, 0, 4)).err(),
        Some(CarveError::SizeMismatch),
        "`from_carve` rejects a zero-length block outright rather than wrapping \
         a pointer it cannot have checked"
    );

    assert!(
        Buf::<f32>::from_bytes(&mut raw.0[0..0]).is_ok(),
        "while `from_bytes` accepts it and yields an empty buf. The asymmetry is \
         real: one constructor has a live `Carve` to interrogate, the other only \
         has a slice that happens to be empty. Pinned rather than smoothed over, \
         because removing either half is an API change."
    );
}

#[test]
fn a_zero_sized_element_is_never_used_as_a_divisor() {
    let mut raw = Aligned::<64>([0; 64]);

    // `()` has size 0. Dividing by it would be a division by zero, so the
    // constructors have to short-circuit; this is the test for that branch.
    let zero_sized = Buf::<()>::from_bytes(&mut raw.0[0..24]).expect("a ZST never divides");

    assert_eq!(zero_sized.len(), 0);
    assert!(zero_sized.is_empty());

    let also_zero = Buf::<()>::from_slice(&mut []);
    assert_eq!(also_zero.len(), 0);
}

#[test]
fn from_bytes_applies_the_same_two_refusals_as_from_carve() {
    let mut raw = Aligned::<64>([0; 64]);

    assert_eq!(
        Buf::<f32>::from_bytes(&mut raw.0[0..18]).err(),
        Some(CarveError::SizeMismatch),
        "18 is not a whole number of f32, while 20 below is — which is why the \
         divisibility case needs a length that is genuinely not a multiple"
    );
    assert_eq!(
        Buf::<f32>::from_bytes(&mut raw.0[1..25]).err(),
        Some(CarveError::Misaligned),
        "24 divides by 4 but the address is one byte into an 8-aligned buffer"
    );

    let good = Buf::<f32>::from_bytes(&mut raw.0[0..24]).expect("24 aligned bytes");
    assert_eq!(good.len(), 6);
    assert_eq!(
        Buf::<f32>::from_bytes(&mut raw.0[0..20]).expect("20 is exactly five f32").len(),
        5,
        "a length that merely looks odd is fine when it does divide; the check \
         is divisibility, not roundness"
    );
}

#[test]
fn from_slice_is_a_view_and_not_a_copy() {
    let mut values = [0.0f32; 4];

    // The buf holds the `&mut` for as long as it lives, so it has to be out of
    // scope before the caller's array can be read. That is the borrow checker
    // refusing to let both views exist at once, which is the point.
    {
        let mut buf = Buf::from_slice(&mut values);

        assert_eq!(buf.len(), 4);

        buf.as_mut_slice()[2] = 7.5;
        buf.as_mut_slice()[0] = -1.0;

        assert_eq!(
            buf.as_slice()[2],
            7.5,
            "reading through the buf sees the write through it"
        );
    }

    assert_eq!(
        values[2], 7.5,
        "and the write is still in the caller's array afterwards, so \
         `from_slice` adopted a view rather than copying"
    );
    assert_eq!(values[0], -1.0);
}

#[test]
fn splitting_in_the_middle_hands_back_the_two_halves() {
    let mut values = [0.0f32; 4];
    let mut buf = Buf::from_slice(&mut values);

    let (left, right) = buf.split_at_mut(2).expect("2 is within a 4-element buf");

    assert_eq!(left.len(), 2);
    assert_eq!(right.len(), 2);

    left[0] = 1.0;
    right[0] = 9.0;

    assert_eq!(values[0], 1.0, "the first half maps to the front");
    assert_eq!(values[2], 9.0, "the second half maps to the back");
    assert_eq!(values[1], 0.0, "and the two halves do not overlap");
}

#[test]
fn splitting_at_either_end_is_allowed_and_yields_one_empty_half() {
    let mut values = [0.0f32; 4];
    let mut buf = Buf::from_slice(&mut values);

    let (left, right) = buf.split_at_mut(0).expect("0 is a valid split point");
    assert_eq!(left.len(), 0);
    assert_eq!(right.len(), 4);

    let (left, right) = buf.split_at_mut(4).expect("len is a valid split point");
    assert_eq!(left.len(), 4);
    assert_eq!(right.len(), 0);
}

#[test]
fn a_split_point_past_the_end_is_refused() {
    let mut values = [0.0f32; 4];
    let mut buf = Buf::from_slice(&mut values);

    assert!(
        buf.split_at_mut(5).is_none(),
        "5 does not exist in a 4-element buf, and returning halves anyway \
         would let the caller name an element the buffer does not have"
    );

    assert_eq!(buf.len(), 4, "a refused split changes nothing");
}

#[test]
fn splitting_past_the_end_of_an_empty_buf_is_still_refused() {
    let mut buf = Buf::<u8>::from_slice(&mut []);

    assert!(
        buf.split_at_mut(1).is_none(),
        "0 is the only split point an empty buf has; answering Some for 1 is \
         what would let `right` claim a byte that was never there"
    );
    assert!(buf.split_at_mut(0).is_some());
}
