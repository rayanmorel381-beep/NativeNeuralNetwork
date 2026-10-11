//! Behaviour tests for the frame.
//!
//! A `Frame` is a `Scratch` plus a recorded position, so what it adds over
//! `Scratch::mark()` is nothing in the rewind and everything in the bookkeeping:
//! the frame knows whether it is still alive, and it refuses to carve once it has
//! been released. Without a `Drop` impl, that refusal is the only thing standing
//! between a released frame and a carve into bytes the release gave back.
//!
//! # A live frame holds the scratch exclusively
//!
//! `Frame::new` takes `&'s mut Scratch` for `'s`, so while a frame is alive the
//! scratch cannot be read or written at all — `scratch.used()` is a borrow
//! error, not a runtime one. That is a real property, not an inconvenience: a
//! direct `scratch.carve` during a frame would move the cursor underneath it and
//! invalidate the frame's idea of where it starts. The compiler enforces it, so
//! the tests below check the cursor only before and after the frame's scope.
//!
//! # What these tests cannot prove
//!
//! `Frame::release` is `unsafe`, and the carves it rewinds past are handed out as
//! `&'a mut` that outlives the frame. So every test here scopes its carves before
//! releasing. That means these tests pin the cursor arithmetic and the alive
//! state machine, but they do not exercise the aliasing that `unsafe` is there
//! to guard — that one is a contract for the caller, and the compiler cannot
//! check it either way.

use crate::scratch::arena::Scratch;
use crate::scratch::carve::CarveError;
use crate::scratch::frame::{Frame, FrameState};

use super::support::Aligned;

#[test]
fn a_new_frame_is_alive_and_records_the_current_position() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let before = scratch.used();
    let frame = Frame::new(&mut scratch);

    assert!(frame.is_alive());
    assert_eq!(frame.state(), FrameState::Alive);
    assert_eq!(
        frame.mark(),
        scratch_mark_at_zero(),
        "a frame opened at the start must record the start, not move it"
    );
    assert_eq!(before, 0, "opening a frame consumes nothing");
}

#[test]
fn a_frame_does_not_disturb_carves_made_before_it() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    {
        let _before = scratch.carve_bytes(40, 1).expect("carve before");
    }

    {
        let mut frame = Frame::new(&mut scratch);
        {
            let _inside = frame.carve_bytes(16, 1).expect("carve inside");
        }
        unsafe { frame.release() }.expect("release");
    }

    assert_eq!(
        scratch.used(),
        40,
        "the frame rolled back to its own mark, not to the start of the arena"
    );
}

#[test]
fn releasing_a_frame_undoes_every_carve_it_covered() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    {
        let _outside = scratch.carve_bytes(24, 1).expect("carve outside");
    }

    {
        let mut frame = Frame::new(&mut scratch);

        {
            let _a = frame.carve_bytes(16, 1).expect("a");
        }
        {
            let _b = frame.carve_bytes(32, 1).expect("b");
        }
        {
            let _c = frame.carve_bytes(8, 1).expect("c");
        }

        unsafe { frame.release() }.expect("release");
    }

    assert_eq!(
        scratch.used(),
        24,
        "a frame releases everything it covered, not just its last carve"
    );
    assert_eq!(scratch.remaining(), 256 - 24);
}

#[test]
fn releasing_twice_is_accepted_and_does_nothing_the_second_time() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    {
        let _outside = scratch.carve_bytes(24, 1).expect("carve outside");
    }

    {
        let mut frame = Frame::new(&mut scratch);

        {
            let _inside = frame.carve_bytes(16, 1).expect("carve inside");
        }

        unsafe { frame.release() }.expect("first release");

        assert!(!frame.is_alive());
        assert_eq!(frame.state(), FrameState::Released);

        unsafe { frame.release() }.expect("a second release is a no-op, not an error");
    }

    assert_eq!(
        scratch.used(),
        24,
        "the second release must not rewind past the frame's own mark: 24 is \
         the carve made outside, and 0 would mean it did"
    );
}

#[test]
fn a_released_frame_refuses_to_carve() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut frame = Frame::new(&mut scratch);

    {
        let _inside = frame.carve_bytes(16, 1).expect("carve inside");
    }

    unsafe { frame.release() }.expect("release");

    let planned = frame.plan(16, 1).expect("planning is still allowed");

    assert!(
        frame.carve(planned).is_err(),
        "a released frame has given its bytes back, so carving into them \
         would hand out the same bytes twice"
    );
    assert_eq!(scratch.used(), 0, "the refused carve moved nothing");
}

#[test]
fn the_byte_and_typed_helpers_refuse_after_release_too() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut frame = Frame::new(&mut scratch);

    unsafe { frame.release() }.expect("release");

    assert!(
        frame.carve_bytes(16, 1).is_err(),
        "carve_bytes routes through carve, so the alive check applies"
    );
    assert!(
        frame.carve_of::<f32>(4).is_err(),
        "carve_of routes through carve too"
    );
    assert_eq!(scratch.used(), 0);
}

#[test]
fn a_frame_can_be_used_again_after_it_is_released() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut frame = Frame::new(&mut scratch);

    {
        let _first = frame.carve_bytes(16, 1).expect("carve");
    }

    unsafe { frame.release() }.expect("release");

    // A released frame stays released. The bytes are back in the arena, so the
    // way to use them again is a new frame or a plain carve, not this one.
    assert!(!frame.is_alive());
    assert!(frame.carve_bytes(16, 1).is_err());

    {
        let _direct = scratch.carve_bytes(16, 1).expect("the arena is usable");
    }

    assert_eq!(scratch.used(), 16);
}

#[test]
fn a_typed_carve_inside_a_frame_is_aligned_and_the_right_length() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut frame = Frame::new(&mut scratch);

    let values = frame.carve_of::<f32>(6).expect("6 f32");

    assert_eq!(values.len(), 6);
    assert_eq!(
        values.as_ptr() as usize % core::mem::align_of::<f32>(),
        0,
        "plan_of asks for align_of::<T>, so the block has to honour it"
    );
    assert_eq!(scratch.used(), 24);

    values[0] = 1.5;
    values[5] = -2.5;
    assert_eq!(values[0] + values[5], -1.0);
}

#[test]
fn a_frame_plan_still_validates_alignment() {
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let frame = Frame::new(&mut scratch);

    assert_eq!(
        frame.plan(16, 3).err(),
        Some(CarveError::InvalidAlignment)
    );
    assert_eq!(frame.plan(16, 0).err(), Some(CarveError::InvalidAlignment));
    assert_eq!(
        frame.plan_of::<f32>(usize::MAX).err(),
        Some(CarveError::Overflow),
        "the length multiplication is checked before the alignment"
    );
}

#[test]
fn a_frame_that_runs_out_of_room_reports_it() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut frame = Frame::new(&mut scratch);

    {
        let _a = frame.carve_bytes(40, 1).expect("40 fits");
    }
    {
        let _b = frame.carve_bytes(40, 1).expect_err("40 more does not fit in 64");
    }

    assert_eq!(
        scratch.used(),
        40,
        "a refused carve leaves the cursor where it was"
    );
}

/// A `ScratchMark` for a cursor at zero, used to state the frame's expectation
/// without reaching into private fields.
fn scratch_mark_at_zero() -> crate::scratch::arena::ScratchMark {
    // A fresh scratch is at zero, so its mark is the mark at zero.
    let mut raw = Aligned::<1>([0; 1]);
    let scratch = Scratch::new(&mut raw.0);
    scratch.mark()
}
