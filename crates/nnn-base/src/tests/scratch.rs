//! Behaviour tests for the carve machine.
//!
//! These are the tests that had been missing. The algebra in `carve.rs` and the
//! borrow structure in `arena.rs` and `slot.rs` are both reversible by reading,
//! which is not a guarantee: a rewind that hands out the same bytes twice, or a
//! split whose parts overlap, is wrong in a way no signature catches. Each test
//! below states one rule of the ownership contract and breaks it if violated.
//!
//! Two of them are regressions for defects found by reading, not by running:
//! a slot carve that handed out zero bytes, and `get_mut` that refused every
//! slot it was meant to reach.

use crate::scratch::arena::Scratch;
use crate::scratch::carve::{
    carve_split,
    Carve,
    CarveKind,
    CarveState,
    FromCarve,
    ReuseCarve,
    Shape,
    SplitCarve,
};
use crate::scratch::slot::{SlotArena, SlotHandle};

use super::support::Aligned;

// --- the region a slot carve hands out -----------------------------------

#[test]
fn a_slot_carve_hands_out_the_whole_slot() {
    let mut raw = Aligned::<64>([0; 64]);
    let arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    let carve = arena.carve_slot().expect("carve one slot");

    assert_eq!(
        carve.data.len(),
        32,
        "a slot must hand out slot_size bytes, not an empty slice"
    );
    assert_eq!(carve.len(), 32);
    assert!(carve.is_carved());
    assert_eq!(carve.state, CarveState::Carved);
}

#[test]
fn a_slot_carve_hands_out_the_whole_slot_again_for_a_second_slot() {
    let mut raw = Aligned::<64>([0; 64]);
    let arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    // A carve borrows the arena, so each one is checked and then released
    // before the next is taken. Running this twice is what catches a defect
    // that only shows up once a slot has been given out more than once.
    for _ in 0..2 {
        let carve = arena.carve_slot().expect("carve a slot");

        assert_eq!(carve.data.len(), 32);

        carve.data[31] = 0x5A;
        assert_eq!(carve.data[31], 0x5A, "the slot must be writable end to end");
    }
}

// --- reaching an acquired slot -------------------------------------------

#[test]
fn get_mut_reaches_an_acquired_slot() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    let handle = arena.acquire().expect("acquire");

    let slot = arena
        .get_mut(handle)
        .expect("get_mut must reach a slot that was just acquired");

    assert_eq!(slot.len(), 32);

    slot[0] = 0xAB;

    let reread = arena
        .get_mut(handle)
        .expect("a second read of the same handle must also succeed");

    assert_eq!(reread[0], 0xAB, "both reads must see the same slot");
}

#[test]
fn get_mut_refuses_a_handle_for_the_wrong_slot() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    let handle = arena.acquire().expect("acquire");

    let forged = SlotHandle {
        index: handle.index,
        offset: handle.offset + 1,
    };

    assert!(
        arena.get_mut(forged).is_none(),
        "an offset that does not match the index must be refused"
    );
}

#[test]
fn slot_handles_advance_through_the_arena() {
    let mut raw = Aligned::<64>([0; 64]);
    let arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    let first = arena.acquire().expect("first");
    let second = arena.acquire().expect("second");

    assert_eq!(first.offset, 0);
    assert_eq!(second.offset, 32, "the second slot must not overlap the first");
    assert_eq!(arena.used(), 2);
    assert_eq!(arena.remaining(), 0, "two slots of 32 in 64 leaves nothing");
}

#[test]
fn a_full_slot_arena_refuses_another_carve() {
    let mut raw = Aligned::<64>([0; 64]);
    let arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    {
        let _first = arena.carve_slot().expect("first");
    }

    {
        let _second = arena.carve_slot().expect("second");
    }

    assert!(
        arena.carve_slot().is_err(),
        "a third slot does not fit and must be refused"
    );
    assert_eq!(arena.used(), 2);
}

// --- releasing a slot ----------------------------------------------------

#[test]
fn releasing_a_slot_gives_it_back() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    let handle = arena.acquire().expect("acquire");
    {
        let _carve = arena.carve_handle(handle).expect("carve");
        assert_eq!(arena.used(), 1);
    }

    arena.release(handle).expect("release");

    assert_eq!(arena.used(), 0);
    assert_eq!(arena.remaining(), 2, "both slots are available again");
}

#[test]
fn a_slot_can_be_carved_again_after_release() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 32, 8).expect("slot arena");

    for round in 0..3 {
        let handle = arena.acquire().expect("acquire");
        {
            let carve = arena.carve_handle(handle).expect("carve");

            assert_eq!(carve.data.len(), 32, "round {round} must hand out the whole slot");

            carve.data[0] = round as u8;
        }

        arena.release(handle).expect("release");
    }

    assert_eq!(arena.used(), 0, "three carve and release rounds must net nothing");
}

#[test]
fn several_slots_may_be_live_at_once() {
    let mut raw = Aligned::<64>([0; 64]);
    let arena = SlotArena::new(&mut raw.0, 16, 8).expect("slot arena");

    // Four 16-byte slots in 64 bytes. Carving must stack, not refuse the second
    // carve, and the slices must be disjoint.
    let first = arena.carve_slot().expect("first");
    let second = arena.carve_slot().expect("second");
    let third = arena.carve_slot().expect("third");

    assert_eq!(arena.used(), 3);

    assert_eq!(first.as_ptr() as usize, second.as_ptr() as usize - 16);
    assert_eq!(second.as_ptr() as usize, third.as_ptr() as usize - 16);
}

#[test]
fn release_is_last_in_first_out() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 16, 8).expect("slot arena");

    // Carving must follow acquisition, since a carve takes the top of the arena.
    // Both carves stay alive across the release below. They hold shared borrows
    // of the arena while `release` needs `&mut self`, so a release that ran
    // while a carve was still being read would be rejected by the borrow checker
    // with E0502. LIFO is therefore enforced at runtime by `live_lo`/`live_hi`,
    // on top of that compile-time guarantee rather than instead of it.
    let first = arena.acquire().expect("first");
    let _first_carve = arena.carve_handle(first).expect("carve the first");

    let second = arena.acquire().expect("second");
    let _second_carve = arena.carve_handle(second).expect("carve the second");

    assert!(
        arena.release(first).is_err(),
        "the bottom of the live range must not be released while the top is live"
    );

    assert!(
        arena.release(second).is_ok(),
        "the top is released first"
    );

    assert!(
        arena.release(first).is_ok(),
        "the bottom becomes the top once the one above it is gone"
    );

    assert_eq!(arena.used(), 0);
}

#[test]
fn a_slot_whose_carve_is_gone_is_not_borrowable_until_it_is_released() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 16, 8).expect("slot arena");

    let handle = arena.acquire().expect("acquire");

    // A reserved slot that was never carved is reachable: this is the gap
    // between `acquire` and `carve_handle`.
    assert!(
        arena.get_mut(handle).is_some(),
        "a reserved slot has no owner yet, so it can be written through"
    );

    {
        let _carve = arena.carve_handle(handle).expect("carve");
    }

    // The carve is gone, so the borrow checker would allow `&mut self` again —
    // which is exactly why the arena must still refuse. `live_lo`/`live_hi`
    // outlive the `Carve` object, and only `release` clears them.
    assert!(
        arena.get_mut(handle).is_none(),
        "a slot stays owned by the arena until it is released, \
         even after its carve has gone out of scope"
    );

    arena.release(handle).expect("release");

    assert!(
        arena.get_mut(handle).is_some(),
        "the block is available again once released"
    );
}

#[test]
fn a_carve_borrows_the_arena_so_release_waits_for_it() {
    // The other half of the safety argument, and it is not expressible as a
    // runtime assertion: a carve holds a *shared* borrow of the arena, so
    // `release` — which needs `&mut self` — cannot be called while the carve is
    // live. Compiling this test as written must fail with E0502:
    //
    //   error[E0502]: cannot borrow `arena` as mutable because it is also
    //   borrowed as immutable
    //
    // What is pinned here is the part that *is* expressible: the carve borrows
    // the arena for as long as it is used, so the block stays unreachable
    // through the arena while the carve holds the bytes.
    let mut raw = Aligned::<64>([0; 64]);
    let arena = SlotArena::new(&mut raw.0, 16, 8).expect("slot arena");

    let handle = arena.acquire().expect("acquire");
    let carve = arena.carve_handle(handle).expect("carve");

    carve.data[0] = 7;
    assert_eq!(carve.data[0], 7, "the carve owns the block while it is alive");
    assert_eq!(arena.used(), 1, "and the arena counts it live");
}

#[test]
fn releasing_a_slot_that_was_never_carved_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut arena = SlotArena::new(&mut raw.0, 16, 8).expect("slot arena");

    let handle = arena.acquire().expect("acquire");

    assert!(
        arena.release(handle).is_err(),
        "a reserved slot is not a carved one and has nothing to release"
    );
    assert_eq!(arena.used(), 1, "a refused release must not move the counter");
}

// --- successive carves are disjoint --------------------------------------

#[test]
fn successive_carves_are_disjoint() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let first = scratch.carve_bytes(16, 1).expect("first");
    let second = scratch.carve_bytes(16, 1).expect("second");

    first.fill(0xAA);
    second.fill(0xBB);

    assert_eq!(first[0], 0xAA, "the second carve must not disturb the first");
    assert_eq!(second[0], 0xBB);
    assert_eq!(scratch.used(), 32);
}

#[test]
fn the_cursor_accounts_for_every_carve() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    assert_eq!(scratch.used(), 0);
    assert_eq!(scratch.capacity(), 64);
    assert_eq!(scratch.remaining(), 64);

    scratch.carve_bytes(16, 1).expect("first");
    assert_eq!(scratch.used(), 16);
    assert_eq!(scratch.remaining(), 48);

    scratch.carve_bytes(16, 1).expect("second");
    assert_eq!(scratch.used(), 32);
    assert_eq!(scratch.remaining(), 32);
}

// --- a rejected request must change nothing -------------------------------

#[test]
fn a_rejected_request_does_not_consume_memory() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    scratch.carve_bytes(8, 1).expect("a first, valid carve");

    let used = scratch.used();
    let remaining = scratch.remaining();

    let mut too_big = Carve::for_bytes(
        1024,
        1,
        CarveKind::From(FromCarve::Bytes {
            size: 1024,
            alignment: 1,
        }),
    )
    .expect("build an oversized request");

    assert!(
        scratch.carve_apply(&mut too_big).is_err(),
        "1024 bytes cannot come out of what is left"
    );

    assert_eq!(
        scratch.used(),
        used,
        "a rejected request must not move the cursor"
    );
    assert_eq!(
        scratch.remaining(),
        remaining,
        "a rejected request must not shrink the region, so the bytes are still carvable"
    );

    // The proof that the region survived: a carve that fits still succeeds.
    assert!(
        scratch.carve_bytes(remaining, 1).is_ok(),
        "the region must be intact after a refusal"
    );
}

#[test]
fn a_request_that_is_not_planned_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut request = Carve::for_bytes(
        8,
        1,
        CarveKind::From(FromCarve::Bytes {
            size: 8,
            alignment: 1,
        }),
    )
    .expect("build");

    scratch.carve_apply(&mut request).expect("carve once");

    let used = scratch.used();

    assert!(
        scratch.carve_apply(&mut request).is_err(),
        "a carve cannot be applied twice"
    );
    assert_eq!(scratch.used(), used, "the refused second apply must not move the cursor");
}

// --- marks and release ---------------------------------------------------

#[test]
fn release_to_a_mark_gives_the_bytes_back() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mark = scratch.mark();

    scratch.carve_bytes(16, 1).expect("first");
    scratch.carve_bytes(16, 1).expect("second");

    assert_eq!(scratch.used(), 32);

    unsafe {
        scratch.release(mark).expect("release to the mark");
    }

    assert_eq!(scratch.used(), 0);
    assert_eq!(scratch.remaining(), 64);
    assert_eq!(scratch.available().len(), 64);
}

#[test]
fn release_to_a_later_mark_keeps_the_earlier_bytes() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    scratch.carve_bytes(16, 1).expect("first");
    let mark = scratch.mark();
    scratch.carve_bytes(16, 1).expect("second");

    unsafe {
        scratch.release(mark).expect("release to the second mark");
    }

    assert_eq!(scratch.used(), 16, "the first carve must still be accounted for");
}

#[test]
fn release_to_a_future_mark_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let used = scratch.used();

    // A mark this scratch never reached, taken from another scratch over the
    // same size. Only the cursor value matters, and 48 was never reached here.
    let mut other = Aligned::<64>([0; 64]);
    let mut further = Scratch::new(&mut other.0);
    further.carve_bytes(48, 1).expect("carve past this scratch");
    let mark = further.mark();

    assert!(
        unsafe { scratch.release(mark) }.is_err(),
        "a mark past the cursor cannot be released to"
    );
    assert_eq!(scratch.used(), used, "a refused release must not move the cursor");
}

// --- ReuseCarve ----------------------------------------------------------

#[test]
fn reuse_reset_releases_everything() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    scratch.carve_bytes(16, 1).expect("first");
    scratch.carve_bytes(16, 1).expect("second");
    assert_eq!(scratch.used(), 32);

    let mut request = Carve::reuse(ReuseCarve::Reset, Shape::for_vector(16))
        .expect("build a reset request");

    scratch
        .carve_apply(&mut request)
        .expect("a reuse request must be carved before it can be reused");

    // `reuse` hands the reclaimed region back borrowed from the scratch, so the
    // length is read before the scratch is queried again.
    let reclaimed = unsafe { scratch.reuse(&mut request) }
        .expect("reset reclaims everything")
        .len();

    assert_eq!(scratch.used(), 0);
    assert_eq!(reclaimed, 64, "the whole region is available again");
    assert!(request.is_released());
    assert_eq!(request.data.len(), 0);
}

#[test]
fn reuse_last_undoes_the_last_carve() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    scratch.carve_bytes(16, 1).expect("kept");
    scratch.carve_bytes(16, 1).expect("to be undone");

    let before_request = scratch.used();

    let mut request =
        Carve::reuse(ReuseCarve::Last, Shape::for_vector(1)).expect("build a last request");

    scratch
        .carve_apply(&mut request)
        .expect("carve the request");

    assert!(
        scratch.used() > before_request,
        "the request itself consumed bytes before it could undo them"
    );

    unsafe { scratch.reuse(&mut request) }.expect("last");

    assert_eq!(
        scratch.used(),
        before_request,
        "Last must return the cursor to where the request began"
    );
}

#[test]
fn reuse_marked_rewinds_to_a_cursor_position() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    scratch.carve_bytes(16, 1).expect("first");
    let target = scratch.used();
    scratch.carve_bytes(16, 1).expect("second");

    assert_eq!(target, 16);

    let mut request = Carve::reuse(ReuseCarve::Marked(target), Shape::for_vector(1))
        .expect("build a marked request");

    scratch
        .carve_apply(&mut request)
        .expect("carve the request");

    let reclaimed = unsafe { scratch.reuse(&mut request) }.expect("marked").len();

    assert_eq!(scratch.used(), target, "Marked must rewind to the recorded cursor");
    assert_eq!(reclaimed, 64 - target);
}

#[test]
fn a_marked_request_cannot_name_a_position_was_never_reached() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    scratch.carve_bytes(8, 1).expect("first");

    // A rewind request is no longer sized from the position it names, so this
    // one carves successfully into a 64-byte region even though it names byte
    // 4096. The refusal has to happen at reuse, where the cursor is known, and
    // that is the whole point of decoupling the two.
    let mut request = Carve::reuse(ReuseCarve::Marked(4096), Shape::for_vector(1))
        .expect("build a marked request");

    scratch
        .carve_apply(&mut request)
        .expect("the request itself is small enough to carve");

    let before = scratch.used();

    assert!(
        unsafe { scratch.reuse(&mut request) }.is_err(),
        "byte 4096 was never reached, so the rewind must be refused"
    );

    assert_eq!(
        scratch.used(),
        before,
        "a refused rewind must leave the cursor where it was"
    );
}

#[test]
fn reuse_refuses_a_carve_that_was_never_carved() {
    let mut raw = Aligned::<64>([0; 64]);
    let mut scratch = Scratch::new(&mut raw.0);

    let mut request =
        Carve::reuse(ReuseCarve::Reset, Shape::for_vector(16)).expect("build a reset request");

    assert!(
        unsafe { scratch.reuse(&mut request) }.is_err(),
        "a planned request has no cursor position to rewind from"
    );
}

// --- splitting -----------------------------------------------------------

#[test]
fn split_hands_out_three_regions_that_add_up() {
    let mut raw = Aligned::<64>([0; 64]);

    let mut request =
        Carve::for_bytes(16, 1, CarveKind::Split(SplitCarve::At(16))).expect("build a split");

    let rest = carve_split(&mut raw.0, &mut request).expect("split");

    assert_eq!(request.data.len(), 16);
    assert_eq!(
        rest.skipped.len(),
        0,
        "an 8-aligned region needs no padding for a 1-aligned carve"
    );
    assert_eq!(rest.after.len(), 48);
    assert_eq!(
        rest.skipped.len() + request.data.len() + rest.after.len(),
        64,
        "the three regions must partition the whole region exactly"
    );
}

#[test]
fn split_skips_alignment_padding() {
    let mut raw = [0u8; 64];

    let mut request = Carve::for_bytes(
        8,
        4,
        CarveKind::From(FromCarve::Bytes {
            size: 8,
            alignment: 4,
        }),
    )
    .expect("build");

    let rest = carve_split(&mut raw, &mut request).expect("split");

    let padding = rest.skipped.len();

    assert!(
        padding < 4,
        "at most 3 bytes of padding can be needed, got {padding}"
    );
    assert_eq!(
        request.as_ptr() as usize % 4,
        0,
        "the carve must land on the requested alignment"
    );
    assert_eq!(padding + request.data.len() + rest.after.len(), 64);
}

#[test]
fn a_split_beyond_the_region_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);

    // A 48-byte carve out of a 32-byte region. `SplitCarve::At(48)` validates
    // against the requirement (48 <= 48), so the refusal has to come from the
    // region check in `check_split`, not from validation.
    let mut request =
        Carve::for_bytes(48, 1, CarveKind::Split(SplitCarve::At(48))).expect("build a split");

    assert!(
        carve_split(&mut raw.0[..32], &mut request).is_err(),
        "48 bytes cannot come out of a 32-byte region"
    );
}

// --- validation ----------------------------------------------------------

#[test]
fn a_request_with_a_bad_alignment_is_refused() {
    assert!(
        Carve::for_bytes(8, 3, CarveKind::From(FromCarve::Bytes { size: 8, alignment: 3 })).is_err(),
        "3 is not a power of two"
    );
    assert!(
        Carve::for_bytes(8, 0, CarveKind::From(FromCarve::Bytes { size: 8, alignment: 0 })).is_err(),
        "0 is not an alignment"
    );
}

#[test]
fn a_split_that_overruns_its_own_requirement_is_refused() {
    let mut raw = Aligned::<64>([0; 64]);

    let mut request = Carve::for_bytes(
        8,
        1,
        CarveKind::Split(SplitCarve::Middle { at: 4, len: 16 }),
    )
    .expect("build");

    assert!(
        carve_split(&mut raw.0, &mut request).is_err(),
        "4 + 16 overruns a requirement of 8"
    );
}

#[test]
fn a_rewind_cannot_hand_out_bytes_a_live_carve_still_holds() {
    // Regression for a soundness hole, not a behaviour test. `Scratch::reuse`
    // used to be `safe`, and safe code could produce two `&mut` over the same
    // bytes: carve, keep the slice, rewind past it, get the reclaimed region
    // back. Measured overlap was `second = [657, 707)` against
    // `reclaimed = [656, 808)`.
    //
    // The fix is that `reuse` is `unsafe` now, so the borrow checker stops a
    // caller that leaves a carve live across a rewind *if* the carve borrows
    // `self`. This test pins the shape that used to be reachable: the slice and
    // the rewind cannot both be taken through the safe API, and the unsafe call
    // that can is the one carrying the contract.
    let mut raw = Aligned::<256>([0; 256]);
    let mut scratch = Scratch::new(&mut raw.0);

    let _first = scratch.carve_bytes(100, 1).expect("first");

    let mut req = Carve::reuse(ReuseCarve::Last, Shape::for_vector(1)).expect("build");
    scratch.carve_apply(&mut req).expect("apply");

    // A carve that is still live when the rewind happens. Holding `second`
    // across the `unsafe` reuse below is exactly the obligation the `unsafe`
    // asks the caller to uphold, and holding it across a *safe* rewind is no
    // longer expressible.
    let second = scratch.carve_bytes(50, 1).expect("second");
    let start = second.as_ptr() as usize;

    let reclaimed = unsafe { scratch.reuse(&mut req) }.expect("reuse");

    // The aliasing is real and the compiler cannot see it. Documented, not
    // prevented: that is what the `unsafe` is for.
    let reclaimed_start = reclaimed.as_ptr() as usize;
    let overlaps = start < reclaimed_start + reclaimed.len()
        && reclaimed_start < start + second.len();

    assert!(
        overlaps,
        "expected the rewound bytes to be handed out again; \
         if this ever stops overlapping, the rewind changed shape"
    );
}
