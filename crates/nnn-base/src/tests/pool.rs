//! Behaviour tests for the size-class pool.
//!
//! A `Pool` is a `Scratch` with a rounding policy bolted on, and the rounding is
//! the whole point: a request is served from the smallest class that holds it, and
//! the block handed out is the *class* capacity, never the requested length. That
//! is what makes blocks interchangeable, and it also means a pool over-consumes:
//! asking for 20 bytes out of a `[16, 32, 64]` table spends 32.
//!
//! The tests below pin that behaviour deliberately, including the two places
//! where the implementation and its own documentation disagree.
//!
//! # `mark()` records the position it is called at
//!
//! A `PoolMark` is not "the state before the last carve". It is the cursor as it
//! stands when `mark()` is called, so releasing to it undoes the work that came
//! *after* it. Two consequences, both pinned below:
//!
//! - A mark taken after the final carve releases nothing, because the rewind
//!   target is where the cursor already is.
//! - There is no mark for the very beginning of the pool. `PoolMark` carries the
//!   class of the last carve, and `last_class` starts as `None`, so `mark()`
//!   returns `None` until something has been carved. "Rewind to the start" is
//!   `reset`, not a mark.

use crate::scratch::carve::CarveError;
use crate::scratch::pool::Pool;

use super::support::{Aligned, CLASSES};

#[test]
fn the_smallest_class_that_holds_the_request_is_chosen() {
    let mut raw = Aligned::<512>([0; 512]);
    let pool = Pool::new(&mut raw.0, CLASSES);

    assert_eq!(pool.capacity_class(1), Some(0), "class 0 is 16 bytes");
    assert_eq!(pool.capacity_class(16), Some(0), "an exact fit takes the class");
    assert_eq!(pool.capacity_class(17), Some(1), "17 does not fit in 16");
    assert_eq!(pool.capacity_class(32), Some(1));
    assert_eq!(pool.capacity_class(33), Some(2));
    assert_eq!(pool.capacity_class(64), Some(2));
    assert_eq!(pool.capacity_class(65), Some(3));
}

#[test]
fn a_request_larger_than_every_class_is_refused() {
    let mut raw = Aligned::<512>([0; 512]);
    let pool = Pool::new(&mut raw.0, CLASSES);

    assert_eq!(pool.capacity_class(129), None, "129 exceeds the largest class");

    assert_eq!(
        pool.plan_bytes(129, 1).err(),
        Some(CarveError::EmptyClasses),
        "the refusal has to name the class table, not the arena"
    );
}

#[test]
fn a_class_of_zero_capacity_is_skipped() {
    let mut raw = Aligned::<128>([0; 128]);
    // A zero entry is a hole, not a class. `capacity_class` filters it out
    // rather than treating it as a block of nothing.
    let pool = Pool::new(&mut raw.0, &[0, 16, 32]);

    assert_eq!(pool.capacity_class(1), Some(1), "class 0 is zero and skipped");
    assert_eq!(pool.capacity_class(16), Some(1));
    assert_eq!(pool.capacity_class(17), Some(2));
}

#[test]
fn a_carve_hands_out_the_class_capacity_and_not_the_request() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    let carve = pool.carve_bytes(20, 1).expect("20 bytes fits class 32");

    assert_eq!(
        carve.data.len(),
        32,
        "a block is the class size, which is what makes it interchangeable"
    );
    assert_eq!(
        pool.used(),
        32,
        "and the cursor moves by the class size, so the 12 spare bytes are spent"
    );
    assert_eq!(pool.remaining(), 512 - 32);
}

#[test]
fn planning_does_not_consume_anything() {
    let mut raw = Aligned::<512>([0; 512]);
    let pool = Pool::new(&mut raw.0, CLASSES);

    let _planned = pool.plan_bytes(20, 1).expect("planning must succeed");

    assert_eq!(pool.used(), 0, "a plan is a description, not an allocation");
    assert_eq!(pool.remaining(), 512);
}

#[test]
fn a_direct_class_carve_uses_that_class_capacity() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    let carve = pool.carve_class(2).expect("class 2 is 64 bytes");

    assert_eq!(carve.data.len(), 64);
    assert_eq!(pool.used(), 64);
    assert_eq!(pool.class_capacity(2), Some(64));
}

#[test]
fn a_direct_class_carve_of_an_empty_class_is_refused() {
    let mut raw = Aligned::<128>([0; 128]);
    let mut pool = Pool::new(&mut raw.0, &[0, 16]);

    assert_eq!(
        pool.carve_class(0).err(),
        Some(CarveError::EmptyClasses),
        "class 0 has no capacity, so there is nothing to carve"
    );
    assert_eq!(pool.carve_class(9).err(), Some(CarveError::EmptyClasses));
    assert_eq!(pool.used(), 0, "a refused carve moves nothing");
}

#[test]
fn a_mark_records_the_position_it_is_taken_at() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    {
        let _first = pool.carve_bytes(20, 1).expect("carve");
    }

    assert_eq!(pool.used(), 32);

    // The mark is taken here, so it names cursor 32 — *after* the first block.
    // Releasing to it therefore undoes whatever comes next, not the first
    // block. A mark taken after the only carve releases nothing at all.
    let mark = pool.mark().expect("a mark exists once something was carved");

    {
        let _second = pool.carve_bytes(20, 1).expect("carve");
    }

    assert_eq!(pool.used(), 64, "two 32-byte blocks");

    unsafe { pool.release(mark) }.expect("release");

    assert_eq!(
        pool.used(),
        32,
        "the second block was 32 bytes, and that is what comes back"
    );
    assert_eq!(pool.remaining(), 512 - 32);
}

#[test]
fn releasing_to_a_mark_taken_after_every_carve_is_a_no_op() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    {
        let _first = pool.carve_bytes(20, 1).expect("carve");
    }

    let mark = pool.mark().expect("mark");
    let used = pool.used();

    unsafe { pool.release(mark) }.expect("release");

    assert_eq!(
        pool.used(),
        used,
        "a mark taken at the current position rewinds to that same position"
    );
}

#[test]
fn there_is_no_mark_before_anything_is_carved() {
    let mut raw = Aligned::<512>([0; 512]);
    let pool = Pool::new(&mut raw.0, CLASSES);

    assert!(
        pool.mark().is_none(),
        "a pool has no class to record yet, so it cannot name a mark at \
         position 0. This is a real restriction of the current design: \
         `last_class` is what a `PoolMark` carries, and it starts as None."
    );
}

#[test]
fn releasing_restores_the_class_so_a_later_mark_is_valid() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    {
        let _first = pool.carve_bytes(20, 1).expect("class 1");
    }

    // The mark names cursor 32. More work happens, then the rewind discards it.
    let mark = pool.mark().expect("mark");

    {
        let _second = pool.carve_bytes(70, 1).expect("class 3");
    }

    assert_eq!(pool.used(), 160);

    unsafe { pool.release(mark) }.expect("release");
    assert_eq!(pool.used(), 32, "rewound to the position the mark named");

    // The point of the test: `release` restored the class rather than clearing
    // it. If it had left `last_class` as None, the mark below would fail and the
    // pool would be unusable for the rest of its life.
    let later = pool.mark().expect("a mark is available after a release");

    {
        let _third = pool.carve_bytes(20, 1).expect("class 1");
    }

    assert_eq!(pool.used(), 64, "carving works again after the rewind");

    unsafe { pool.release(later) }.expect("release");

    assert_eq!(pool.used(), 32);
}

#[test]
fn reset_gives_everything_back_and_forgets_the_class() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    {
        let _a = pool.carve_bytes(20, 1).expect("carve");
    }
    {
        let _b = pool.carve_bytes(20, 1).expect("carve");
    }

    assert_eq!(pool.used(), 64);

    unsafe { pool.reset() };

    assert_eq!(pool.used(), 0);
    assert!(
        pool.mark().is_none(),
        "reset clears the class, so the pool goes back to having no mark"
    );
}

#[test]
fn a_typed_carve_multiplies_before_choosing_a_class() {
    let mut raw = Aligned::<512>([0; 512]);
    let mut pool = Pool::new(&mut raw.0, CLASSES);

    // 5 f32 = 20 bytes, so class 32 — the multiplication has to happen before
    // the class is picked, or the pool would round the element count instead.
    let values = pool.carve::<f32>(5).expect("5 f32");

    assert_eq!(values.len(), 5);
    assert_eq!(pool.used(), 32, "still a 32-byte block");
}

#[test]
fn an_alignment_that_is_not_a_power_of_two_is_refused() {
    let mut raw = Aligned::<512>([0; 512]);
    let pool = Pool::new(&mut raw.0, CLASSES);

    assert_eq!(
        pool.plan_bytes(32, 3).err(),
        Some(CarveError::InvalidAlignment)
    );
    assert_eq!(
        pool.plan_bytes(32, 0).err(),
        Some(CarveError::InvalidAlignment)
    );
}

#[test]
fn with_alignment_falls_back_to_one_for_a_bad_alignment() {
    let mut raw = Aligned::<512>([0; 512]);

    let pool = Pool::with_alignment(&mut raw.0, CLASSES, 16);
    assert_eq!(pool.alignment(), 16);

    // A bad alignment is silently replaced rather than refused, so a caller that
    // passes 24 gets a working pool with 1-byte alignment instead of an error.
    let pool = Pool::with_alignment(&mut raw.0, CLASSES, 24);
    assert_eq!(pool.alignment(), 1);

    let pool = Pool::with_alignment(&mut raw.0, CLASSES, 0);
    assert_eq!(pool.alignment(), 1);
}

#[test]
fn a_class_table_that_is_not_ascending_selects_the_first_fit_not_the_smallest() {
    let mut raw = Aligned::<512>([0; 512]);
    let pool = Pool::new(&mut raw.0, &[64, 16, 32]);

    // `capacity_class` scans in table order and returns the first entry that
    // fits. The module documentation says it "picks the smallest class that can
    // hold the request", which is only true of an ascending table — and `new`
    // does not check the order. For a 20-byte request this picks 64 where the
    // smallest fitting class is 32.
    //
    // Pinned rather than fixed: making the pool reject an unordered table would
    // mean adding an error variant, which is an API change and not a test fix.
    assert_eq!(pool.capacity_class(20), Some(0), "first fit, not smallest fit");
    assert_eq!(pool.class_capacity(0), Some(64));
}
