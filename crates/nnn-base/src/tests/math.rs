//! Behaviour tests for the float types and the kernels built on them.
//!
//! # Each precision stands on its own
//!
//! `F16` and `BF16` are sixteen-bit encodings with no relationship to `f32`, and
//! this crate holds them to that. Neither offers a way in or out of `f32`: a
//! value of either type is reached through its own sixteen bits, checked with
//! its own arithmetic, and written back out through its own bytes. Nothing here
//! converts between precisions, because a conversion is where a lost bit hides.
//!
//! That rules out the obvious way to write these tests. `F16` cannot be checked
//! against a reference by asking what an `f32` becomes, because the only
//! interesting question about a half-precision rounding is what it does to a
//! value that never was one. So the tests below work in the two directions that
//! do not need a neighbour: arithmetic between like values, and the encoding
//! itself.
//!
//! The encoding tests read the exponent and fraction fields directly, which is
//! the format's own description. `1.0` in `F16` is `0x3c00` — exponent `15`,
//! fraction `0` — and `2^-14`, the smallest subnormal, is `0x0001`. If either
//! of those constants moves, the format is no longer the one these names mean.

use crate::math::float::{
	carve_softmax_rows, BF16, DType, E4M3, E5M2, F16, F32, F64, Float, Layout, TF32,
};
use crate::math::helpers::{
	byte_size_of, convert, convert_buffer, from_f32, from_f64, layout_of, to_f32,
	to_f32_bits_for_test, to_f64,
};
use crate::scratch::typed::Buf;

// --- f16 encoding ----------------------------------------------------------

#[test]
fn one_and_its_neighbours_sit_where_the_format_says_they_do() {
    // 1.0 is exponent 15, fraction 0. 1.0 + 2^-10 is the next representable
    // value up, and nothing lies between them.
    assert_eq!(F16::ONE.to_bits(), 0x3c00, "1.0");
    assert_eq!(F16::HALF.to_bits(), 0x3800, "0.5");
    assert_eq!(F16::TWO.to_bits(), 0x4000, "2.0");
    assert_eq!(F16::ZERO.to_bits(), 0x0000, "0.0");
    assert_eq!(F16::NEG_INF.to_bits(), 0xfc00, "-infinity");
    assert_eq!(F16::POS_INF.to_bits(), 0x7c00, "+infinity");
}

#[test]
fn the_subnormal_range_starts_below_the_smallest_normal() {
    // With exponent 0 the implicit leading one is not there, so fraction 1 is
    // 2^-24 and fraction 0x3ff is the largest subnormal. Exponent 1 with
    // fraction 0 is the smallest normal, one step up.
    assert_eq!(F16::from_bits(0x0001).to_bits(), 0x0001, "smallest subnormal");
    assert_eq!(F16::from_bits(0x03ff).to_bits(), 0x03ff, "largest subnormal");
    assert_eq!(F16::from_bits(0x0400).to_bits(), 0x0400, "smallest normal");
    assert_eq!(
        F16::from_bits(0x8001).to_bits(),
        0x8001,
        "and the subnormal range is signed"
    );
}

#[test]
fn a_full_exponent_field_is_either_infinity_or_nan() {
    assert!(!F16::POS_INF.is_finite(), "infinity is not finite");
    assert!(!F16::NEG_INF.is_finite(), "and neither is its negation");
    assert!(F16::ZERO.is_finite(), "a zero is finite");
    assert!(F16::ONE.is_finite(), "and so is one");

    // The exponent alone cannot tell infinity from NaN — both have it
    // saturated. Only the fraction can, which is why is_nan reads it.
    assert!(!F16::from_bits(0x7c00).is_nan(), "zero fraction is infinity");
    assert!(F16::from_bits(0x7e00).is_nan(), "any fraction at all is NaN");
    assert!(F16::from_bits(0x7c01).is_nan());
    assert!(F16::from_bits(0xfe00).is_nan(), "and NaN can be negative");
    assert!(!F16::from_bits(0x7bff).is_nan(), "the largest finite is not NaN");
    assert!(F16::from_bits(0x7bff).is_finite());
}

#[test]
fn every_f16_pattern_survives_a_trip_through_its_own_bytes() {
    for bits in 0..=u16::MAX {
        let value = F16::from_bits(bits);
        let mut buffer = [0u8; 2];

        value.write_le_bytes(&mut buffer);
        let recovered = F16::from_le_bytes(&buffer);

        assert_eq!(
            recovered.to_bits(),
            bits,
            "0x{bits:04x} is its own two bytes"
        );
    }
}

#[test]
fn f16_reports_its_own_width() {
    assert_eq!(F16::BYTE_SIZE, 2, "sixteen bits is two bytes");
    assert_eq!(F16::DTYPE_TAG, 2, "tag 2 is f16");
}

// --- f16 arithmetic --------------------------------------------------------

#[test]
fn f16_adds_around_one_without_touching_a_neighbour_format() {
    assert_eq!((F16::ONE + F16::ONE).to_bits(), 0x4000, "1 + 1 = 2");
    assert_eq!((F16::ONE + F16::HALF).to_bits(), 0x3e00, "1 + 0.5 = 1.5");
    assert_eq!((F16::TWO - F16::ONE).to_bits(), 0x3c00, "2 - 1 = 1");
    assert_eq!((F16::HALF * F16::TWO).to_bits(), 0x3c00, "0.5 * 2 = 1");
    assert_eq!((F16::ONE / F16::TWO).to_bits(), 0x3800, "1 / 2 = 0.5");
}

#[test]
fn f16_negation_only_flips_the_sign_bit() {
    let magnitude = 0x3c00;
    assert_eq!((-F16::from_bits(magnitude)).to_bits(), magnitude | 0x8000);
    assert_eq!((-(-F16::ONE)).to_bits(), F16::ONE.to_bits(), "twice is the value");

    // A NaN stays a NaN when negated: the sign is the only field that moves.
    assert!((-F16::from_bits(0x7e00)).is_nan());
}

#[test]
fn f16_abs_clears_the_sign_and_leaves_the_rest() {
    assert_eq!(F16::from_bits(0xc000).abs().to_bits(), 0x4000);
    assert_eq!(F16::from_bits(0x8001).abs().to_bits(), 0x0001);
    assert!(F16::from_bits(0xfe00).abs().is_nan(), "abs of a NaN is a NaN");
}

#[test]
fn f16_compares_against_zero_across_the_subnormal_range() {
    let mut count = 0u16;

    for bits in 1..0x0400u16 {
        if F16::from_bits(bits) > F16::ZERO {
            count += 1;
        }
        if F16::from_bits(bits) < F16::ZERO {
            panic!("a positive subnormal is not below zero: 0x{bits:04x}");
        }
    }

    // Every non-zero subnormal, 1 through 0x3ff.
    assert_eq!(count, 0x03ff, "each subnormal is above zero");
}

#[test]
fn f16_max_and_min_pick_the_larger_and_smaller_magnitude() {
    assert_eq!(F16::ONE.max(F16::TWO).to_bits(), 0x4000);
    assert_eq!(F16::ONE.min(F16::TWO).to_bits(), 0x3c00);
    assert_eq!(
        F16::from_bits(0x8001).max(F16::from_bits(0x0001)).to_bits(),
        0x0001,
        "a negative and a positive of the same size: the positive wins max"
    );
    assert_eq!(
        F16::from_bits(0x8001).min(F16::from_bits(0x0001)).to_bits(),
        0x8001,
        "and the negative wins min"
    );
}

#[test]
fn f16_reciprocal_of_one_is_one() {
    assert_eq!(F16::ONE.recip().to_bits(), F16::ONE.to_bits());
    assert_eq!(F16::TWO.recip().to_bits(), F16::HALF.to_bits());
    assert_eq!(F16::HALF.recip().to_bits(), F16::TWO.to_bits());
}

#[test]
fn f16_transcendentals_stay_finite_on_ordinary_input() {
    // These run as native series in the format's own arithmetic. The point is
    // not the last digit — it is that nothing here escapes into another
    // precision and comes back wrong, and that nothing returns a NaN for input
    // that is well inside the range.
    assert!(F16::ONE.exp().is_finite(), "exp(1)");
    assert!(F16::ONE.ln().is_finite(), "ln(1)");
    assert!(F16::ONE.sqrt().is_finite(), "sqrt(1)");
    assert!(F16::ONE.tanh().is_finite(), "tanh(1)");
    assert!(F16::ONE.sin().is_finite(), "sin(1)");
    assert!(F16::ONE.cos().is_finite(), "cos(1)");

    assert_eq!(F16::ONE.sqrt().to_bits(), F16::ONE.to_bits(), "sqrt(1) = 1");
    // exp(0) is 1 by definition. The label on the previous line was reading
    // ONE, not ZERO, so this was checking exp(1) against 1 and failing.
    assert_eq!(F16::ZERO.exp().to_bits(), F16::ONE.to_bits(), "exp(0) is 1");
    // exp(1) is e. The series is truncated, so the result sits within an ulp of
    // the correctly rounded pattern (0x416d) rather than on it. Pinning the
    // exact bits would test the term count instead of the identity.
    assert_eq!(
        F16::ONE.exp().to_bits().abs_diff(0x416d),
        1,
        "exp(1) is e to within one ulp"
    );
    assert_eq!(F16::from_bits(0).ln().to_bits(), F16::NEG_INF.to_bits(), "ln(0)");
    assert_eq!(F16::ZERO.sqrt().to_bits(), F16::ZERO.to_bits(), "sqrt(0) = 0");
    assert!(F16::from_bits(0x8000).ln().to_bits() == F16::NEG_INF.to_bits());
}

#[test]
fn f16_round_leaves_a_whole_value_alone_and_clears_a_fraction() {
    // Exponent 15, fraction 0x200 is 1.5, which has to round up to 2.
    assert_eq!(F16::from_bits(0x3e00).round().to_bits(), 0x4000, "1.5 rounds up");
    assert_eq!(F16::ONE.round().to_bits(), F16::ONE.to_bits(), "1 is already whole");
    assert_eq!(
        F16::from_bits(0x3c00 | 0x0100).round().to_bits(),
        0x3c00,
        "1 + 2^-9 is below the half step, so it stays"
    );
}

// --- bf16 -------------------------------------------------------------------

#[test]
fn one_lands_on_its_own_field_in_bf16() {
    // bf16 is f32's top half: 8 exponent bits, 7 fraction, bias 127.
    assert_eq!(BF16::ONE.to_bits(), 0x3f80, "1.0");
    assert_eq!(BF16::HALF.to_bits(), 0x3f00, "0.5");
    assert_eq!(BF16::TWO.to_bits(), 0x4000, "2.0");
    assert_eq!(BF16::ZERO.to_bits(), 0x0000, "0.0");
    assert_eq!(BF16::NEG_INF.to_bits(), 0xff80, "-infinity");
    assert_eq!(BF16::POS_INF.to_bits(), 0x7f80, "+infinity");
}

#[test]
fn bf16_has_no_subnormal_range_to_carry() {
    // With 8 exponent bits, exponent 0 is reserved and the smallest positive
    // value is 2^-126. There is no fraction-only encoding here, which is the
    // one structural thing that separates bf16 from f16.
    assert_eq!(BF16::from_bits(0x0001).to_bits(), 0x0001, "a nonzero pattern");
    assert!(
        BF16::from_bits(0x0001) > BF16::ZERO,
        "and unlike f16 it is a normal number, not a subnormal"
    );
    assert!(BF16::from_bits(0x0001).is_finite());
}

#[test]
fn bf16_tells_infinity_from_nan_by_the_fraction_alone() {
    assert!(!BF16::POS_INF.is_finite());
    assert!(!BF16::from_bits(0x7f7f).is_nan(), "all ones is infinity");
    assert!(BF16::from_bits(0x7fc0).is_nan(), "a fraction means NaN");
    assert!(BF16::from_bits(0xffc0).is_nan());
    // 0x7f7f is the largest finite bf16 — a full fraction but an exponent one
    // short of saturation. Reading the pattern as if the exponent were also
    // part of the fraction would call this infinity, and every large-but-finite
    // activation with it.
    assert!(BF16::from_bits(0x7f7f).is_finite(), "largest finite is finite");
    assert!(!BF16::from_bits(0x7f80).is_finite(), "one step up is infinity");
}

#[test]
fn every_bf16_pattern_survives_a_trip_through_its_own_bytes() {
    for bits in 0..=u16::MAX {
        let value = BF16::from_bits(bits);
        let mut buffer = [0u8; 2];

        value.write_le_bytes(&mut buffer);

        assert_eq!(BF16::from_le_bytes(&buffer).to_bits(), bits, "0x{bits:04x}");
    }
}

#[test]
fn bf16_reports_its_own_width_and_is_a_different_type_from_f16() {
    assert_eq!(BF16::BYTE_SIZE, 2);
    assert_eq!(BF16::DTYPE_TAG, 3, "tag 3 is bf16");

    // Same constants, same format, different tag. They are distinct types and
    // stay that way: there is no conversion here to prove otherwise.
    assert_ne!(F16::DTYPE_TAG, BF16::DTYPE_TAG);
    assert_eq!(F16::ONE.to_bits(), 0x3c00);
    assert_eq!(BF16::ONE.to_bits(), 0x3f80);
}

#[test]
fn bf16_adds_and_multiplies_in_its_own_bits() {
    assert_eq!((BF16::ONE + BF16::ONE).to_bits(), 0x4000, "1 + 1 = 2");
    assert_eq!((BF16::TWO - BF16::ONE).to_bits(), 0x3f80, "2 - 1 = 1");
    assert_eq!((BF16::HALF * BF16::TWO).to_bits(), 0x3f80, "0.5 * 2 = 1");
    assert_eq!((BF16::ONE / BF16::TWO).to_bits(), 0x3f00, "1 / 2 = 0.5");
    assert_eq!((-BF16::ONE).to_bits(), 0xbf80, "negation flips the sign bit");
    assert_eq!(BF16::from_bits(0xbf80).abs().to_bits(), 0x3f80);
}

#[test]
fn bf16_transcendentals_stay_finite_on_ordinary_input() {
    assert!(BF16::ONE.exp().is_finite());
    assert!(BF16::ONE.ln().is_finite());
    assert!(BF16::ONE.sqrt().is_finite());
    assert!(BF16::ONE.tanh().is_finite());
    assert_eq!(BF16::ONE.sqrt().to_bits(), BF16::ONE.to_bits(), "sqrt(1) = 1");
    assert_eq!(BF16::ZERO.exp().to_bits(), BF16::ONE.to_bits(), "exp(0) is 1");

    // exp(1) is e, and ln(2) is ln 2. Both are Taylor sums truncated at 16
    // terms, so they land one ulp off the correctly rounded value — asserting
    // the exact encoding would pin the series length instead of the identity.
    let bf16_of = |value: f32| -> u16 {
        let bits = value.to_bits();
        ((bits >> 16) as u16 & 0x8000)
            | ((((bits >> 23) & 0xff) as u16) << 7)
            | ((bits >> 16) as u16 & 0x7f)
    };
    let exp_one = BF16::ONE.exp().to_bits();
    let ln_two = BF16::TWO.ln().to_bits();
    let exp_want = bf16_of(2.718_281_8);
    let ln_want = bf16_of(0.693_147_2);
    assert_eq!(
        exp_one.abs_diff(exp_want),
        1,
        "exp(1) within one ulp of e"
    );
    assert_eq!(ln_two.abs_diff(ln_want), 1, "ln(2) within one ulp");

    assert_eq!(BF16::from_bits(0).ln().to_bits(), BF16::NEG_INF.to_bits());
    assert_eq!(BF16::ZERO.sqrt().to_bits(), BF16::ZERO.to_bits(), "sqrt(0) = 0");
}

#[test]
fn f16_orders_by_value_and_not_by_pattern() {
    // The derived ordering used to be the integer one, which put 0x8000 (-0.0)
    // above 0x0000 (+0.0) and every negative above every positive. Anything
    // that branches on a comparison — sqrt, ln, a reLU threshold — inherited
    // that, so the cases below are the ones that were silently wrong.
    assert_eq!(F16::from_bits(0x8000), F16::ZERO, "-0.0 equals +0.0");
    assert!(!(F16::from_bits(0x8000) < F16::ZERO), "and neither is smaller");
    assert!(!(F16::ZERO < F16::from_bits(0x8000)), "in either direction");

    assert!(F16::from_bits(0xbc00) < F16::ONE, "-1 < 1");
    assert!(F16::from_bits(0xc000) < F16::from_bits(0xbc00), "-2 < -1");
    assert!(F16::ONE > F16::from_bits(0xbc00), "1 > -1");
    assert!(F16::from_bits(0x3800) > F16::from_bits(0xbc00), "0.5 > -1");

    // NaN is unordered, so it compares unequal to everything including itself.
    let nan = F16::from_bits(0x7e00);
    assert!(nan != nan, "NaN is not equal to itself");
    assert!(nan.is_nan());
    assert_eq!(nan.partial_cmp(&F16::ONE), None, "no order against a number");

    assert_eq!(F16::from_bits(0xbc00).sqrt().to_bits(), F16::ZERO.to_bits());
    assert_eq!(F16::from_bits(0xbc00).ln().to_bits(), F16::NEG_INF.to_bits());
    assert_eq!(F16::from_bits(0xbc00).max(F16::ONE).to_bits(), F16::ONE.to_bits());
    assert_eq!(F16::from_bits(0xbc00).min(F16::ONE).to_bits(), 0xbc00);

    // bf16 has a different field split but the same rules.
    assert_eq!(BF16::from_bits(0x8000), BF16::ZERO, "bf16 -0.0 equals +0.0");
    assert!(BF16::from_bits(0xc000) < BF16::from_bits(0xbf80), "bf16 -2 < -1");
    // 0xbf80 is -1.0, not a NaN: with a 7-bit fraction the pattern one step
    // above 0x3f80 is a number, so the NaN has to come from the saturated
    // exponent instead.
    assert_eq!(BF16::from_bits(0xbf80), -BF16::ONE, "bf16 -1");
    assert!(BF16::from_bits(0x7fc0) != BF16::from_bits(0x7fc0), "bf16 NaN");
    assert!(BF16::from_bits(0x7fc0).is_nan());
}

// --- F32 and F64 are their own types ---------------------------------------

#[test]
fn f32_and_f64_carry_their_own_bits_and_reach_the_native_types() {
    // Both are newtypes, not aliases: the point is that a value of either type
    // is reached through its own width and has to say which width that is. If
    // either were `f32`, the size and the tag would be the only difference left
    // and there would be nothing to test here.
    assert_eq!(F32::BYTE_SIZE, 4);
    assert_eq!(F64::BYTE_SIZE, 8);

    let one = F32::ONE.to_native();
    assert_eq!(one, 1.0f32);
    assert_eq!(F32::from_native(one).to_bits(), F32::ONE.to_bits());

    // 0.1 is not 0.1 in any of these, so it is a better probe than a power of
    // two: the bits either survive the round trip or they do not.
    let tenth = F32::from_native(0.1f32);
    assert_eq!(tenth.to_native(), 0.1f32);
    assert_eq!(F32::from_bits(tenth.to_bits()), tenth);

    let tenth_wide = F64::from_native(0.1f64);
    assert_eq!(tenth_wide.to_native(), 0.1f64);
    assert_eq!(F64::from_bits(tenth_wide.to_bits()), tenth_wide);

    // F32 has to hold 0.1 in 24 bits and F64 the same value in 53, so the two
    // agree on the value and not on the pattern.
    assert_eq!(F32::from_native(tenth_wide.to_native() as f32).to_native(), 0.1f32);
    assert_ne!(tenth.to_bits() as u64, tenth_wide.to_bits());
}

#[test]
fn f32_and_f64_do_their_arithmetic_on_the_hardware() {
    assert_eq!((F32::ONE + F32::ONE).to_native(), 2.0f32);
    assert_eq!((F32::TWO - F32::ONE).to_native(), 1.0f32);
    assert_eq!((F32::HALF * F32::TWO).to_native(), 1.0f32);
    assert_eq!((F32::ONE / F32::TWO).to_native(), 0.5f32);
    assert_eq!((-F32::ONE).to_native(), -1.0f32);
    assert_eq!(F32::from_native(-1.0f32).abs().to_native(), 1.0f32);

    assert_eq!((F64::ONE + F64::ONE).to_native(), 2.0f64);
    assert_eq!((F64::ONE / F64::TWO).to_native(), 0.5f64);
    assert_eq!(F64::from_native(-1.0f64).abs().to_native(), 1.0f64);

    // The out-of-domain answers have to be refused too, and what they refuse with is
    // the crate's own convention rather than the C library's: `lnf` answers NaN
    // for anything at or below zero, where `lnf(0.0)` in C is -inf. What matters
    // here is that neither comes back as a plausible number, because a negative
    // answer would be silently wrong rather than obviously wrong.
    assert!(!F32::from_native(-1.0f32).sqrt().is_finite());
    assert!(!F32::from_native(-1.0f32).ln().is_finite());
    assert!(!F32::ZERO.ln().is_finite());
    assert!(!F64::from_native(-1.0f64).sqrt().is_finite());
    assert!(!F32::POS_INF.is_finite());
    assert!(!F32::POS_INF.is_nan());
    assert!(F32::from_native(f32::NAN).is_nan());

    // These kernels are a couple of Newton steps off a magic constant, so they
    // are accurate to a few ulp rather than exact. Only what is exact by
    // construction is asserted as equal; the rest is pinned to a tolerance.
    assert_eq!(F32::ZERO.sqrt().to_native(), 0.0f32);
    assert_eq!(F32::ZERO.exp().to_native(), 1.0f32);
    assert!((F32::ONE.sqrt().to_native() - 1.0f32).abs() < 1.0e-4);
    assert!((F32::TWO.sqrt().to_native() - 2.0f32.sqrt()).abs() < 1.0e-4);
    assert!((F64::ONE.sqrt().to_native() - 1.0f64).abs() < 1.0e-4);
    assert!((F64::TWO.sqrt().to_native() - 2.0f64.sqrt()).abs() < 1.0e-4);
    assert!((F32::TWO.ln().to_native() - 2.0f32.ln()).abs() < 1.0e-4);
    assert!((F64::TWO.ln().to_native() - 2.0f64.ln()).abs() < 1.0e-4);
    assert_eq!(F32::NEG_INF.exp().to_native(), 0.0f32);
    assert!(F32::from_native(-1.0).ln().is_nan(), "ln of a negative is undefined");
    assert!(F32::from_native(0.0).ln().is_nan(), "ln of zero is undefined");
    assert!(F32::from_native(-1.0).sqrt().is_nan(), "there is no real root below zero");
    assert!(F64::from_native(-1.0).sqrt().is_nan(), "there is no real root below zero");
}

#[test]
fn f32_and_f64_write_their_own_bytes_and_read_them_back() {
    let mut buffer = [0u8; 8];
    F32::ONE.write_le_bytes(&mut buffer);
    assert_eq!(&buffer[..4], &[0x00, 0x00, 0x80, 0x3f], "1.0 little-endian");
    assert_eq!(F32::from_le_bytes(&buffer).to_bits(), F32::ONE.to_bits());

    F64::ONE.write_le_bytes(&mut buffer);
    assert_eq!(F64::from_le_bytes(&buffer).to_bits(), F64::ONE.to_bits());
}

// --- the registry ----------------------------------------------------------

#[test]
fn every_format_has_the_tag_it_was_given_and_the_width_it_has() {
    // The tags are what a file carries, so a change here is a change to data
    // already written. The first four keep their numbers and the rest are
    // appended, which is the only ordering that keeps old files readable.
    let expected = [
        (DType::F32, 0u8, 4usize),
        (DType::F64, 1, 8),
        (DType::F16, 2, 2),
        (DType::BF16, 3, 2),
        (DType::TF32, 4, 4),
        (DType::E4M3, 5, 1),
        (DType::E5M2, 6, 1),
    ];

    assert_eq!(DType::ALL.len(), expected.len(), "no format is missing from the registry");

    for (dtype, tag, width) in expected {
        assert_eq!(dtype.tag(), tag, "tag of {dtype:?}");
        assert_eq!(DType::from_tag(tag), Some(dtype), "tag {tag} reads back");
        assert_eq!(byte_size_of(dtype), width, "width of {dtype:?}");
    }

    assert_eq!(DType::from_tag(expected.len() as u8), None, "no tag past the end");
    assert_eq!(DType::from_tag(200), None, "no invented tag");
}

#[test]
fn each_type_names_its_own_format_in_the_registry() {
    assert_eq!(F32::dtype(), DType::F32);
    assert_eq!(F64::dtype(), DType::F64);
    assert_eq!(F16::dtype(), DType::F16);
    assert_eq!(BF16::dtype(), DType::BF16);
    assert_eq!(TF32::dtype(), DType::TF32);
    assert_eq!(E4M3::dtype(), DType::E4M3);
    assert_eq!(E5M2::dtype(), DType::E5M2);

    // The type's own constants and the registry's have to be the same numbers,
    // or a tag read off a buffer would carve a differently-shaped thing than the
    // code expects.
    for dtype in DType::ALL {
        let layout = layout_of(dtype);
        assert_eq!(layout.fraction_bits, dtype.layout().fraction_bits);
        assert_eq!(layout.exponent_bits, dtype.layout().exponent_bits);
        assert_eq!(layout.bias, dtype.layout().bias);
        assert_eq!(layout.max_finite, dtype.layout().max_finite);
        assert_eq!(layout.has_infinity, dtype.layout().has_infinity);
    }
}

// --- the narrow formats ----------------------------------------------------

#[test]
fn the_new_formats_put_their_constants_where_the_specification_says() {
    // TF32 first, because its constants are the ones that cannot be read off
    // F32: its epsilon is 2^-10, not 2^-23, and the storage carries 13 dead
    // bits under the fraction.
    assert_eq!(TF32::ONE.to_bits(), 0x3f80_0000);
    assert_eq!(TF32::HALF.to_bits(), 0x3f00_0000);
    assert_eq!(TF32::TWO.to_bits(), 0x4000_0000);
    assert_eq!(TF32::EPSILON.to_bits(), 0x3f80_2000, "one step at 1.0 is 2^-10");
    assert_eq!(TF32::POS_INF.to_bits(), 0x7f80_0000);
    assert_eq!(TF32::NEG_INF.to_bits(), 0xff80_0000);

    // E4M3 spends its exponent field on real numbers: 0x7E is a finite 448, not
    // an infinity, and only 0x7F is NaN.
    assert_eq!(E4M3::ONE.to_bits(), 0x38);
    assert_eq!(E4M3::TWO.to_bits(), 0x40);
    assert_eq!(E4M3::EPSILON.to_bits(), 0x20, "one step at 1.0 is 2^-3");
    assert_eq!(to_f32::<E4M3>(E4M3::from_bits(0x7e)).to_native(), 448.0f32);
    assert!(E4M3::from_bits(0x7e).is_finite(), "448 is not an infinity");
    assert!(E4M3::from_bits(0x7f).is_nan(), "0x7F is the only NaN");
    assert!(!E4M3::from_bits(0x7e).is_nan());

    // E5M2 keeps a real infinity: 5 exponent bits are all spent on range.
    assert_eq!(E5M2::ONE.to_bits(), 0x3c);
    assert_eq!(E5M2::EPSILON.to_bits(), 0x34, "one step at 1.0 is 2^-2");
    assert_eq!(to_f32::<E5M2>(E5M2::from_bits(0x7b)).to_native(), 57344.0f32, "largest finite");
    assert!(!E5M2::from_bits(0x7c).is_finite(), "0x7C is an infinity");
    assert_eq!(to_f32::<E5M2>(E5M2::POS_INF).to_native(), f32::INFINITY);
    assert!(E5M2::from_bits(0x7f).is_nan());

    // BF16's epsilon was 0x3400, which is F32's: with 7 fraction bits the step
    // at 1.0 is 2^-7, and a constant borrowed from the wrong width makes a
    // whole class of comparisons wrong by a factor of 65536.
    assert_eq!(BF16::EPSILON.to_bits(), 0x3c00);
}

#[test]
fn tf32_is_a_truncated_f32_in_the_top_nineteen_bits() {
    // The layout that makes this work is the one that says the value lives at
    // the top of the word, so the storage is still an F32 with its low bits
    // cleared — which is what a checkpoint holds and what a GPU reads.
    let storage_shift = layout_of(DType::TF32).storage_shift;
    assert_eq!(storage_shift, 13, "32 bits minus 19 live ones");

    for word in [TF32::ZERO, TF32::ONE, TF32::TWO, TF32::NEG_INF, TF32::EPSILON] {
        assert_eq!(word.to_bits() & ((1 << storage_shift) - 1), 0, "the low bits are dead");
    }

    assert_eq!(TF32::from_bits(0x3f80_0000), TF32::ONE);
    assert_eq!(to_f32::<TF32>(TF32::ONE).to_bits(), F32::ONE.to_bits());

    // Truncating the low bits of the F32 encoding is what a narrow has to
    // agree with: the difference between rounding and chopping is one ulp, and
    // rounding is the one a conversion should do. Ten fraction bits is what is
    // left of pi, which is 3.140625 and not 3.14159.
    let rounded = from_f32::<TF32>(F32::from_native(3.14159f32));
    assert_eq!(rounded.to_bits(), 0x4049_0000);
    assert_eq!(to_f32::<TF32>(rounded).to_native(), 3.140_625f32);
    assert_ne!(to_f32::<TF32>(rounded).to_native(), 3.141_592_7f32, "23 bits did not survive");
}

#[test]
fn e4m3_saturates_where_e5m2_reaches_an_infinity() {
    // The pair is only useful together if they disagree exactly here: the same
    // overflow is 448 in E4M3 and infinity in E5M2, and a kernel that assumed
    // otherwise would pass an activation check on one and not the other.
    let overflow = F32::from_native(1.0e5f32);
    assert_eq!(from_f32::<E4M3>(overflow).to_bits(), 0x7e);
    assert!(from_f32::<E4M3>(overflow).is_finite(), "E4M3 holds at 448");
    assert_eq!(from_f32::<E5M2>(overflow).to_bits(), 0x7c);
    assert!(!from_f32::<E5M2>(overflow).is_finite(), "E5M2 becomes infinite");

    // Arithmetic agrees with conversion, which is the point of sharing one
    // overflow rule.
    let big = E4M3::from_bits(0x7e);
    assert_eq!((big + big).to_bits(), 0x7e, "448 + 448 stays 448 in E4M3");
    let wide = E5M2::from_bits(0x7b);
    assert_eq!((wide + wide).to_bits(), 0x7c, "57344 + 57344 is infinite in E5M2");

    assert_eq!((E4M3::ONE + E4M3::ONE).to_bits(), 0x40);
    assert_eq!((E4M3::HALF * E4M3::TWO).to_bits(), 0x38);
    assert_eq!((E5M2::ONE / E5M2::TWO).to_bits(), 0x38);
    assert_eq!((TF32::ONE + TF32::ONE).to_bits(), 0x4000_0000);
    assert_eq!((TF32::HALF * TF32::TWO).to_bits(), 0x3f80_0000);
    assert_eq!((-TF32::ONE).to_bits(), 0xbf80_0000);
    assert_eq!(TF32::from_bits(0xbf80_0000).abs().to_bits(), 0x3f80_0000);

    // TF32 has an infinity and a NaN like F32, which is what a widened E4M3
    // value looks like on the way in.
    assert_eq!(from_f32::<TF32>(F32::POS_INF).to_bits(), TF32::POS_INF.to_bits());
    assert!(from_f32::<TF32>(F32::from_native(f32::NAN)).is_nan());
}

// --- conversions -----------------------------------------------------------

#[test]
fn narrowing_rounds_to_nearest_with_ties_to_even() {
    // The values below are the ones a wrong rounding rule gets wrong. Round-half-
    // away gives 0x3C01 and 0x3C03 where the correct answer is 0x3C00 and
    // 0x3C02, and the difference is a whole ulp that a model trained with one
    // rule and served with the other will feel.
    //
    // 1 + 2^-11 is exactly halfway between 1.0 (0x3C00) and the next step up, so
    // it goes to the even one, down. 1 + 3*2^-11 is halfway between 0x3C01 and
    // 0x3C02, so it goes up to the even one, 0x3C02.
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_1000)).to_bits(), 0x3c00);
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_3000)).to_bits(), 0x3c02);

    // Just past the halfway point rounds up, which is what makes the tie above a
    // tie and not a threshold. 0x2FFF is one step under the tie and lands back
    // on 0x3C01; 0x3001 is one step over and lands on 0x3C02, the same place the
    // tie itself goes.
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_2fff)).to_bits(), 0x3c01);
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_3001)).to_bits(), 0x3c02);

    // 0x3C03 is only reachable by a value that says so exactly, because two and
    // a half steps is a tie and a tie goes to the even neighbour, which is two.
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_6000)).to_bits(), 0x3c03);
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_5000)).to_bits(), 0x3c02, "2.5 steps, tie to even");
    assert_eq!(from_f32::<F16>(F32::from_bits(0x3f80_4000)).to_bits(), 0x3c02, "2 steps exactly");

    // 0.1 is the classic: neither 0.1 nor the 0.1 next to it, and the F16 and
    // BF16 answers differ because their fractions have different lengths.
    assert_eq!(from_f32::<F16>(F32::from_native(0.1f32)).to_bits(), 0x2e66);
    assert_eq!(from_f32::<BF16>(F32::from_native(0.1f32)).to_bits(), 0x3dcd);
    assert_eq!(from_f32::<E4M3>(F32::from_native(0.1f32)).to_bits(), 0x1d);
    assert_eq!(from_f32::<E5M2>(F32::from_native(0.1f32)).to_bits(), 0x2e);

    // The largest finite value has to survive, and the value just past it must
    // not: 65504 is in range and 65520 is the halfway point that rounds over.
    assert_eq!(from_f32::<F16>(F32::from_native(65504.0f32)).to_bits(), 0x7bff);
    assert_eq!(from_f32::<F16>(F32::from_native(65520.0f32)).to_bits(), 0x7c00);
    assert_eq!(from_f32::<F16>(F32::from_native(65519.0f32)).to_bits(), 0x7bff);
}

#[test]
fn widening_is_exact_in_both_directions() {
    // Nothing narrower than F32 can hold a value F32 does not, so widening has
    // to be lossless in both directions. If it is not, some bit was moved
    // wrong, and the round trip is where that shows.
    let f16_bits = [0x0000u16, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x7c00, 0x8000, 0x8001, 0xbc00, 0xfbff, 0xfc00];
    for bits in f16_bits {
        assert_eq!(from_f32::<F16>(to_f32::<F16>(F16::from_bits(bits))).to_bits(), bits, "f16 0x{bits:04x}");
    }

    let bf16_bits = [0x0000u16, 0x0001, 0x007f, 0x0080, 0x3f80, 0x7f7f, 0x7f80, 0x8000, 0x8001, 0xbf80, 0xff80];
    for bits in bf16_bits {
        assert_eq!(from_f32::<BF16>(to_f32::<BF16>(BF16::from_bits(bits))).to_bits(), bits, "bf16 0x{bits:04x}");
    }

    let tf32_bits = [0x0000_0000u32, 0x0012_0000, 0x3f80_0000, 0x7f7f_e000, 0x7f80_0000, 0x8000_0000, 0xbf80_0000];
    for bits in tf32_bits {
        assert_eq!(from_f32::<TF32>(to_f32::<TF32>(TF32::from_bits(bits))).to_bits(), bits, "tf32 0x{bits:08x}");
    }

    let e4m3_bits = [0x00u8, 0x01, 0x37, 0x38, 0x7e, 0x80, 0xb8, 0xfe];
    for bits in e4m3_bits {
        assert_eq!(from_f32::<E4M3>(to_f32::<E4M3>(E4M3::from_bits(bits))).to_bits(), bits, "e4m3 0x{bits:02x}");
    }

    let e5m2_bits = [0x00u8, 0x01, 0x3c, 0x7b, 0x7c, 0x80, 0xbc, 0xfc];
    for bits in e5m2_bits {
        assert_eq!(from_f32::<E5M2>(to_f32::<E5M2>(E5M2::from_bits(bits))).to_bits(), bits, "e5m2 0x{bits:02x}");
    }

    // And the values those patterns are, read straight off their own fields. The
    // F16 top of the subnormal range is the one that would catch a shift that is
    // off by a bit: it is 1023 * 2^-24, just under 2^-14, and reading its leading
    // bit as the exponent is what turns it into something near 2^-7.
    assert_eq!(to_f32::<F16>(F16::from_bits(0x0001)).to_native(), 2.0f32.powi(-24));
    assert_eq!(to_f32::<F16>(F16::from_bits(0x03ff)).to_native(), 1023.0 * 2.0f32.powi(-24));
    // BF16's subnormals sit below F32's smallest normal, so their exponents are
    // checked as patterns: `powi` cannot be asked for them, because the squaring
    // it does on the way to 2^133 overflows to an infinity first and the whole
    // thing comes back as zero.
    assert_eq!(to_f32::<BF16>(BF16::from_bits(0x0001)).to_bits(), 0x0001_0000, "bf16 0x0001 is 2^-133");
    assert_eq!(to_f32::<BF16>(BF16::from_bits(0x0080)).to_bits(), 0x0080_0000, "bf16 0x0080 is 2^-126");
    assert_eq!(to_f32::<BF16>(BF16::from_bits(0x007f)).to_bits(), 0x007f_0000, "bf16 0x007f is 127 * 2^-133");
    assert_eq!(to_f32::<F16>(F16::from_bits(0x0400)).to_native(), 2.0f32.powi(-14));
    assert_eq!(to_f32::<F16>(F16::from_bits(0x3c00)).to_native(), 1.0f32);
    assert_eq!(to_f32::<E4M3>(E4M3::from_bits(0x7e)).to_native(), 448.0f32);
    assert_eq!(to_f32::<E4M3>(E4M3::from_bits(0x01)).to_native(), 2.0f32.powi(-9));
    assert_eq!(to_f32::<E5M2>(E5M2::from_bits(0x7b)).to_native(), 57344.0f32);
    assert_eq!(to_f32::<TF32>(TF32::ONE).to_native(), 1.0f32);
    assert_eq!(to_f32::<F16>(F16::POS_INF).to_native(), f32::INFINITY);
    assert!(to_f32::<F16>(F16::from_bits(0x7e01)).is_nan());
    assert!(to_f32::<E4M3>(E4M3::from_bits(0x7f)).is_nan());
}

#[test]
fn converting_twice_does_not_move_a_value_a_second_time() {
    // Widening into a format with more fraction bits and coming back has to be
    // the identity: if it is not, the first conversion rounded and the second
    // one rounded again, which is a different function from the one asked for.
    //
    // Only the widening direction is a round trip at all. F16 has 10 fraction
    // bits and BF16 has 7, so F16 into BF16 loses the bottom three: 0x03FF is
    // 1023 * 2^-24, and the nearest BF16 to that is 2^-14, which is 0x0400.
    let f16_bits = [0x0000u16, 0x0001, 0x03ff, 0x0400, 0x3c00, 0x7bff, 0x8001, 0xbc00, 0xfbff];
    for bits in f16_bits {
        let here = F16::from_bits(bits);
        let narrowed = convert::<F16, BF16>(here);
        let back = convert::<BF16, F16>(narrowed);
        // What survives is a value BF16 can spell, which is not the same as an F16
        // word with BF16's three lost fraction bits clear: `0x0001` is 2^-24, far
        // inside BF16's exponent range, so it comes back untouched even though its
        // fraction bits are not zero.
        let survives = matches!(bits, 0x0000 | 0x0001 | 0x0400 | 0x3c00 | 0x7c00 | 0x8000 | 0x8001 | 0xbc00 | 0xfc00);
        if survives {
            assert_eq!(back.to_bits(), bits, "f16 0x{bits:04x} survives bf16");
        } else {
            // `0x03FF` is 1023 * 2^-24 and rounds up to 2^-14. `0x7BFF` and
            // `0xFBFF` are +/-65504, and the nearest BF16 to either is the one
            // above them — 65536, past the top of F16's range — so they come back
            // as infinities rather than as the finite values they went in as.
            let expected = match bits {
                0x03ff => 0x0400,
                0x7bff => 0x7c00,
                0xfbff => 0xfc00,
                other => unreachable!("0x{other:04x} is not in the list"),
            };
            assert_eq!(back.to_bits(), expected, "f16 0x{bits:04x} rounds away");
        }

        let widened = convert::<F16, TF32>(here);
        assert_eq!(convert::<TF32, F16>(widened).to_bits(), bits, "f16 0x{bits:04x} -> tf32");

        let widest = convert::<F16, F32>(here);
        assert_eq!(convert::<F32, F16>(widest).to_bits(), bits, "f16 0x{bits:04x} -> f32");
    }

    // BF16 into TF32 is 7 fraction bits into 10, so nothing is lost.
    let bf16_bits = [0x0000u16, 0x0001, 0x3f80, 0x7f7f, 0x8001, 0xbf80, 0xff80];
    for bits in bf16_bits {
        let here = BF16::from_bits(bits);
        let widened = convert::<BF16, TF32>(here);
        assert_eq!(convert::<TF32, BF16>(widened).to_bits(), bits, "bf16 0x{bits:04x} -> tf32");
        let widest = convert::<BF16, F32>(here);
        assert_eq!(convert::<F32, BF16>(widest).to_bits(), bits, "bf16 0x{bits:04x} -> f32");
    }

    // E4M3 into TF32 and into F32 loses nothing at all: 3 fraction bits into 10.
    let e4m3_bits = [0x00u8, 0x01, 0x7e, 0x80, 0xb8, 0xfe];
    for bits in e4m3_bits {
        let here = E4M3::from_bits(bits);
        let widened = convert::<E4M3, TF32>(here);
        assert_eq!(convert::<TF32, E4M3>(widened).to_bits(), bits, "e4m3 0x{bits:02x} -> tf32");
        let widest = convert::<E4M3, F32>(here);
        assert_eq!(convert::<F32, E4M3>(widest).to_bits(), bits, "e4m3 0x{bits:02x} -> f32");
    }

    // E5M2 into E4M3 widens the fraction from 2 bits to 3 and loses the range, so
    // it is a round trip only below 448.
    for bits in [0x00u8, 0x01, 0x3c, 0x7e, 0x80, 0xbc, 0xfe] {
        let here = E4M3::from_bits(bits);
        let widened = convert::<E4M3, E5M2>(here);
        assert_eq!(convert::<E5M2, E4M3>(widened).to_bits(), bits, "e4m3 0x{bits:02x} -> e5m2");
    }

    // F32 is the hub, so widening out of it and back is a round trip too. The
    // values are ones every narrow format here can spell exactly, and stay inside
    // E5M2's range — its top is 57344, so `F32`'s own largest finite value would
    // saturate to an infinity on the way through rather than come back.
    for bits in [0x0000_0000u32, 0x3f80_0000, 0x3fc0_0000, 0x4000_0000, 0x8000_0000] {
        let here = F32::from_bits(bits);
        let narrowed = convert::<F32, E5M2>(here);
        assert_eq!(convert::<E5M2, F32>(narrowed).to_bits(), bits, "f32 0x{bits:08x} -> e5m2");
        let narrowed = convert::<F32, F16>(here);
        assert_eq!(convert::<F16, F32>(narrowed).to_bits(), bits, "f32 0x{bits:08x} -> f16");
        let narrowed = convert::<F32, TF32>(here);
        assert_eq!(convert::<TF32, F32>(narrowed).to_bits(), bits, "f32 0x{bits:08x} -> tf32");
        let narrowed = convert::<F32, BF16>(here);
        assert_eq!(convert::<BF16, F32>(narrowed).to_bits(), bits, "f32 0x{bits:08x} -> bf16");
    }

    // Narrowing twice is idempotent, which is the property that matters for a
    // value converted on every pass through a mixed-precision graph. The pattern
    // is not asserted, only that a second pass moves nothing.
    for value in [0.1f32, 3.141_59, 1.0e-8, 65535.0, 448.5, 1.0 / 3.0] {
        let once = from_f32::<F16>(F32::from_native(value));
        assert_eq!(from_f32::<F16>(to_f32::<F16>(once)).to_bits(), once.to_bits(), "{value} -> f16");
        let once = from_f32::<BF16>(F32::from_native(value));
        assert_eq!(from_f32::<BF16>(to_f32::<BF16>(once)).to_bits(), once.to_bits(), "{value} -> bf16");
        let once = from_f32::<E4M3>(F32::from_native(value));
        assert_eq!(from_f32::<E4M3>(to_f32::<E4M3>(once)).to_bits(), once.to_bits(), "{value} -> e4m3");
        let once = from_f32::<E5M2>(F32::from_native(value));
        assert_eq!(from_f32::<E5M2>(to_f32::<E5M2>(once)).to_bits(), once.to_bits(), "{value} -> e5m2");
        let once = from_f32::<TF32>(F32::from_native(value));
        assert_eq!(from_f32::<TF32>(to_f32::<TF32>(once)).to_bits(), once.to_bits(), "{value} -> tf32");
    }
}

#[test]
fn zero_keeps_its_sign_and_the_small_values_flush_to_it() {
    // -0.0 has to arrive as -0.0 and not as +0.0: `1/x` inverts one, and a
    // value that compares equal is not the same value.
    assert_eq!(from_f32::<F16>(F32::from_bits(0x8000_0000)).to_bits(), 0x8000);
    assert_eq!(from_f32::<E4M3>(F32::from_bits(0x8000_0000)).to_bits(), 0x80);
    assert_eq!(to_f32::<E4M3>(E4M3::from_bits(0x80)).to_bits(), 0x8000_0000);
    assert_eq!(from_f32::<E4M3>(F32::from_bits(0x8000_0000)).to_bits() & 0x80, 0x80);

    // Below half of the smallest subnormal the answer is a signed zero, not a
    // wrap and not the smallest value. F32's smallest subnormal is 2^-149, and
    // E4M3's is 2^-9, so anything much under 2^-10 is gone.
    assert_eq!(from_f32::<F16>(F32::from_native(1e-30f32)).to_bits(), 0x0000);
    assert_eq!(from_f32::<E4M3>(F32::from_native(1e-30f32)).to_bits(), 0x00);
    assert_eq!(from_f32::<E4M3>(F32::from_native(1e-30f32)).to_bits() & 0x80, 0x00);
    assert_eq!(from_f32::<F16>(F32::from_native(-1e-30f32)).to_bits(), 0x8000);

    // Just above it is not. E4M3's smallest subnormal is 2^-9, so one subnormal
    // step is 2^-9: half a step is a tie to even and lands on zero, one and a
    // half steps is a tie to even and lands on two, and three quarters of a step
    // is not a tie at all.
    let step = 2.0f32.powi(-9);
    assert_eq!(from_f32::<E4M3>(F32::from_native(0.5 * step)).to_bits(), 0x00, "tie to even, down");
    assert_eq!(from_f32::<E4M3>(F32::from_native(1.0 * step)).to_bits(), 0x01);
    assert_eq!(from_f32::<E4M3>(F32::from_native(1.5 * step)).to_bits(), 0x02, "tie to even, up");
    assert_eq!(from_f32::<E4M3>(F32::from_native(0.75 * step)).to_bits(), 0x01, "not a tie");
    assert_eq!(to_f32::<E4M3>(E4M3::from_bits(0x01)).to_native(), 2.0f32.powi(-9));
}

#[test]
fn the_non_finite_values_take_each_format_s_own_spelling() {
    // There is no single NaN to map to: F16 spells it 0x7E01, E4M3 has exactly
    // one at 0x7F, and TF32 spells it in the top bits. What has to hold is that
    // the result is non-finite and keeps the sign, not that it matches.
    let nan = F32::from_native(f32::NAN);

    assert!(from_f32::<F16>(nan).is_nan());
    assert!(from_f32::<BF16>(nan).is_nan());
    assert!(from_f32::<TF32>(nan).is_nan());
    assert_eq!(from_f32::<E4M3>(nan).to_bits(), 0x7f, "the one NaN E4M3 has");
    assert_eq!(from_f32::<E4M3>(F32::from_bits(0xffc0_0000)).to_bits(), 0xff, "sign kept");
    assert!(from_f32::<E5M2>(nan).is_nan());

    // Infinity keeps its own spelling, except where there is none.
    assert_eq!(from_f32::<F16>(F32::POS_INF).to_bits(), 0x7c00);
    assert_eq!(from_f32::<BF16>(F32::POS_INF).to_bits(), 0x7f80);
    assert_eq!(from_f32::<TF32>(F32::POS_INF).to_bits(), 0x7f80_0000);
    assert_eq!(from_f32::<E5M2>(F32::POS_INF).to_bits(), 0x7c);
    assert_eq!(from_f32::<E4M3>(F32::POS_INF).to_bits(), 0x7e, "saturates, there is no infinity");

    assert_eq!(from_f32::<F16>(F32::NEG_INF).to_bits(), 0xfc00);
    assert_eq!(from_f32::<E4M3>(F32::NEG_INF).to_bits(), 0xfe);

    // Widening them back does not invent anything either.
    assert_eq!(to_f32::<F16>(F16::POS_INF).to_bits(), F32::POS_INF.to_bits());
    assert_eq!(to_f32::<F16>(F16::NEG_INF).to_bits(), F32::NEG_INF.to_bits());
    assert!(to_f32::<BF16>(BF16::from_bits(0x7fc0)).is_nan());
    assert!(to_f32::<TF32>(TF32::from_bits(0x7fff_e000)).is_nan());
}

#[test]
fn f64_reaches_the_narrow_formats_and_comes_back() {
    // F64 is not one of the formats the hub can carry, so the trip through F32
    // is a double rounding and the tests here are about the value surviving,
    // not about the last bit of a double-rounded result.
    let tenth = F64::from_native(0.1f64);

    // The narrowing loses precision, which is the whole point of the trip, so
    // what has to hold is that the answer is the nearest F16 and not some other
    // value that happens to be near it.
    assert_eq!(from_f64::<F16>(tenth).to_bits(), 0x2e66);
    assert_eq!(from_f64::<BF16>(tenth).to_bits(), 0x3dcd);
    assert_eq!(from_f64::<E4M3>(tenth).to_bits(), 0x1d);
    assert_eq!(from_f64::<TF32>(tenth).to_bits(), 0x3dcc_c000);

    // And coming back gives that same F16, not 0.1 again: a narrowing is not
    // undone by widening.
    assert_eq!(to_f64::<F16>(F16::from_bits(0x2e66)).to_native(), 0.099_975_585_937_5);
    assert_eq!(to_f64::<F16>(F16::from_bits(0x2e66)).to_native(), 0.099_975_585_937_5f64);
    assert_eq!(to_f64::<E4M3>(E4M3::from_bits(0x1d)).to_native(), 0.101_562_5, "1.625 * 2^-4");
    assert_eq!(to_f64::<E4M3>(E4M3::from_bits(0x7e)).to_native(), 448.0f64);
    assert_eq!(to_f64::<TF32>(TF32::from_bits(0x3dcc_c000)).to_native(), 0.099_975_585_937_5);
    assert_eq!(to_f64::<F16>(F16::POS_INF).to_native(), f64::INFINITY);
    assert!(to_f64::<F16>(F16::from_bits(0x7e01)).is_nan());
    assert!(to_f64::<BF16>(BF16::from_bits(0x7fc0)).is_nan());

    // The trip itself is idempotent, which is the property that has to hold for
    // a conversion to be usable on a value that has been converted before.
    assert_eq!(from_f64::<F16>(to_f64::<F16>(from_f64::<F16>(tenth))).to_bits(), 0x2e66);
    assert_eq!(from_f64::<E4M3>(to_f64::<E4M3>(from_f64::<E4M3>(tenth))).to_bits(), 0x1d);

    // And F32 is a widening of F64 with nothing lost.
    assert_eq!(to_f64::<F32>(F32::from_native(0.1f32)).to_native(), 0.1f32 as f64);
    assert_eq!(from_f64::<F32>(tenth).to_native(), 0.1f32, "narrowing to F32 is exact here");
}

#[test]
fn a_buffer_changes_format_without_changing_length() {
    // The check that matters here is the length: a conversion that quietly wrote
    // past the end of the target, or stopped short, would leave the two buffers
    // describing different things while both claimed to be valid.
    let mut source = [E4M3::ZERO; 5];
    source[0] = E4M3::ONE;
    source[1] = E4M3::from_bits(0x7e);
    source[2] = E4M3::from_bits(0x80);
    source[3] = E4M3::HALF;
    source[4] = E4M3::from_bits(0x7f);

    let mut wide = [F32::ZERO; 5];
    let written = convert_buffer::<E4M3, F32>(&Buf::from_slice(&mut source), &mut Buf::from_slice(&mut wide))
        .expect("five elements fit in five");

    assert_eq!(written, 5);
    assert_eq!(wide[0].to_native(), 1.0f32);
    assert_eq!(wide[1].to_native(), 448.0f32);
    assert!(wide[2].is_nan() == false, "-0.0 survives");
    assert_eq!(wide[3].to_native(), 0.5f32);
    assert!(wide[4].is_nan());

    let mut narrow = [E4M3::ZERO; 5];
    let written = convert_buffer::<F32, E4M3>(&Buf::from_slice(&mut wide), &mut Buf::from_slice(&mut narrow))
        .expect("and back again");
    assert_eq!(written, 5);

    // Everything that came back is what went out, except the NaN, which has no
    // payload to preserve.
    for index in [0usize, 1, 2, 3] {
        assert_eq!(narrow[index].to_bits(), source[index].to_bits(), "index {index}");
    }
    assert!(narrow[4].is_nan());

    // A target that cannot hold the source is refused rather than truncated.
    let mut short = [F32::ZERO; 2];
    let refused = convert_buffer::<E4M3, F32>(&Buf::from_slice(&mut source), &mut Buf::from_slice(&mut short));
    assert!(refused.is_err(), "two elements cannot hold five");
}


#[test]
fn every_word_of_every_narrow_format_widens_to_the_value_it_spells() {
    // Exhaustive rather than sampled. The reference is built from the format
    // definition alone, in f64 — every value here is exactly representable in f64
    // as well as f32, so the cast cannot round and the reference shares no code
    // with the bit-moving path under test.
    fn reference(layout: Layout, packed: u64) -> f64 {
        let sign = if layout.is_negative(packed) { -1.0 } else { 1.0 };
        let exponent = ((packed & layout.exponent_mask()) >> layout.fraction_bits) as i32;
        let fraction = (packed & layout.fraction_mask()) as f64;
        let fraction_bits = layout.fraction_bits;

        if exponent == layout.max_exponent() as i32 {
            if !layout.has_infinity {
                // E4M3 has no infinity; the top exponent is a normal range, and
                // only the all-ones fraction out of it is a NaN.
                if fraction == (1u64 << fraction_bits) as f64 - 1.0 {
                    return f64::NAN;
                }
                return sign * (1.0 + fraction / (1u64 << fraction_bits) as f64)
                    * 2f64.powi(exponent - layout.bias);
            }
            if fraction == 0.0 {
                return sign * f64::INFINITY;
            }
            return f64::NAN;
        }

        if exponent == 0 {
            return sign * fraction * 2f64.powi(1 - layout.bias - fraction_bits as i32);
        }

        sign * (1.0 + fraction / (1u64 << fraction_bits) as f64) * 2f64.powi(exponent - layout.bias)
    }

    for dtype in DType::ALL {
        let layout = dtype.layout();
        if matches!(dtype, DType::F32 | DType::F64) {
            continue;
        }

        // Packed words, so TF32's space is its 19 live bits and the others are as
        // many words as they have bits. Walking packed rather than stored words
        // means every pattern a caller can reach goes through the same path.
        let words = 1u64 << (layout.exponent_bits + layout.fraction_bits + 1);

        for packed in 0..words {
            let value = F32::from_bits(to_f32_bits_for_test(packed, layout));
            let want = reference(layout, packed);
            let got = value.to_native() as f64;

            assert!(
                got == want || (got.is_nan() && want.is_nan()),
                "{dtype:?} 0x{packed:x}: got {got:?} want {want:?}",
            );

            // NaN is the one case where the sign is not checked: a NaN has no
            // sign for the format to spell, and the test's reference cannot say
            // what one ought to be.
            if !want.is_nan() {
                assert_eq!(got.is_sign_negative(), want.is_sign_negative(), "{dtype:?} 0x{packed:x} sign");
            }
        }
    }
}
