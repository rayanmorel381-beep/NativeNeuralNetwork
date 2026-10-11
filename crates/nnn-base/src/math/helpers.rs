use super::elementwise;
use super::float::{BF16, DType, E4M3, E5M2, F16, F32, F64, Float, Layout, MathError, TF32};
use crate::scratch::carve::CarveError;
use crate::scratch::typed::Buf;
use core::marker::PhantomData;

pub(crate) fn abs_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::abs)
}

pub(crate) fn reciprocal_carve<T: Float>(buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
	elementwise::unary(buffer, <T as Float>::recip)
}

pub(crate) fn ldexp_carve<T: Float>(
	buffer: &mut Buf<'_, T>,
	exponent: i32,
) -> Result<usize, MathError> {
	if !(-126..=127).contains(&exponent) {
		return Err(CarveError::Overflow.into());
	}

	// 2^exponent by repeated doubling, never by asking a wider format for the
	// integer exponent. Each squaring is an exact operation on T, so the factor
	// is a native T built from T alone.
	let factor = scale_by_power_of_two::<T>(exponent);

	elementwise::scaled(buffer, factor)
}

/// `2^exponent` for any `T: Float`, by halving and doubling from `T::ONE`.
///
/// A power of two is representable in every IEEE format, so this lands on an
/// exact value for any exponent in range and needs no conversion of `exponent`
/// into the element type. Overflow and underflow land on the format's own
/// infinities and zeros, which is what saturating scale arithmetic should give.
fn scale_by_power_of_two<T: Float>(exponent: i32) -> T {
	let mut factor = T::ONE;
	let mut remaining = exponent.unsigned_abs();
	let mut two = T::TWO;

	while remaining != 0 {
		if remaining & 1 == 1 {
			factor *= two;
		}
		two *= two;
		remaining >>= 1;
	}

	if exponent < 0 {
		T::ONE / factor
	} else {
		factor
	}
}

pub(crate) fn floorf_carve(buffer: &mut Buf<'_, F32>) -> Result<usize, MathError> {
	elementwise::unary(buffer, |value: F32| F32::from_native(floorf(value.to_native())))
}

pub(crate) fn floord_carve(buffer: &mut Buf<'_, F64>) -> Result<usize, MathError> {
	elementwise::unary(buffer, |value: F64| F64::from_native(floord(value.to_native())))
}

pub(crate) fn floorf16_carve(buffer: &mut Buf<'_, F16>) -> Result<usize, MathError> {
	elementwise::unary(buffer, |value: F16| F16::from_bits((f32::from_bits(value.to_bits() as u32)).to_bits() as u16))
}

pub(crate) fn floor_bf16_carve(buffer: &mut Buf<'_, BF16>) -> Result<usize, MathError> {
	elementwise::unary(buffer, |value: BF16| BF16::from_bits((f32::from_bits(value.to_bits() as u32)).to_bits() as u16))
}

#[inline]
pub(crate) fn floord(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let t = x as i64 as f64;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

#[inline]
pub(crate) fn floorf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let t = x as i32 as f32;
    if t > x {
        t - 1.0
    } else {
        t
    }
}

#[inline]
pub(crate) fn ldexpf(x: f32, exp: i32) -> f32 {
    if x == 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    let mant = bits & 0x007f_ffff;
    let mut e = ((bits >> 23) & 0xff) as i32 - 127;
    e += exp;
    if e <= -127 {
        return 0.0;
    }
    if e >= 128 {
        return f32::INFINITY;
    }
    let new_bits = sign | (((e + 127) as u32) << 23) | mant;
    f32::from_bits(new_bits)
}

// ---------------------------------------------------------------------------
// Conversions
//
// Every conversion goes through `F32`, and that is a deliberate limit rather
// than a shortcut. A value that exists in one narrow format and not in another
// has to be rounded, and rounding is the only step here that can lose
// information; doing it in `F32` means the rule is the hardware's rule, written
// once, instead of a second rounding rule per pair of formats.
//
// The one thing that makes this cheap is that the bias does not appear in the
// arithmetic. A real number's exponent is a property of the number, not of the
// format spelling it, so both sides agree on it and the bias only decides which
// bits the exponent field ends up in. That leaves the fraction as the only
// thing to move, and how far it moves is the difference in fraction bits alone:
// `23 - fraction_bits`, or `24 - unbiased - bias - fraction_bits` once the value
// is small enough to need a subnormal on the far side.
// ---------------------------------------------------------------------------

/// `value >> shift`, rounded to nearest with ties to even.
///
/// A shift of zero drops nothing, and a shift wider than `value` drops all of
/// it — both are ordinary cases here, because `F32` is on one side of every one
/// of these conversions and a value can be far below the target's smallest
/// subnormal.
fn round_shift_right(value: u64, shift: i32) -> u64 {
	if shift <= 0 {
		return value << (-shift) as u32;
	}

	if shift >= 65 {
		return 0;
	}

	let shift = shift as u32;
	let kept = value >> shift;
	let dropped = value & ((1u64 << shift) - 1);
	let half = 1u64 << (shift - 1);
	kept + u64::from(dropped > half || (dropped == half && kept & 1 == 1))
}

/// The `F32` pattern a packed word of `layout` widens to, exactly.
///
/// Every format below 32 bits has no more fraction and no more exponent than
/// `F32`, so nothing is dropped here — the fraction only has to move up, and a
/// subnormal only has to find its leading bit so the exponent has something to
/// say. The non-finite cases cannot be widened by moving bits, since each format
/// spells them differently, so they are spelled out.
fn to_f32_bits(word: u64, layout: Layout) -> u32 {
	let sign = ((word & layout.sign_mask()) as u32) << (31 - layout.exponent_bits - layout.fraction_bits);
	let magnitude = word & layout.magnitude_mask();

	if magnitude == 0 {
		return sign;
	}

	if layout.is_nan_word(word) {
		// One canonical F32 NaN, keeping the sign: the payload of the source is
		// not portable, and a second one would not compare equal anyway.
		return sign | 0x7fc0_0000;
	}

	if layout.exponent_saturated(word) {
		return sign | f32::INFINITY.to_bits();
	}

	let exponent = ((magnitude & layout.exponent_mask()) >> layout.fraction_bits) as i32;
	let fraction = magnitude & layout.fraction_mask();
	let fraction_bits = layout.fraction_bits as i32;

	// Reduce the word to a 24-bit significand under F32's implicit bit, plus the
	// unbiased exponent it hangs off, so that the value is `significand * 2^exponent`.
	//
	// A normal has the implicit one to give it an exponent of its own. A
	// subnormal does not, and its highest set bit is what supplies one: `fraction`
	// is `2^leading` or more and less than `2^(leading+1)`, which puts the value
	// in the binade that bit names.
	let (significand, exponent) = if exponent != 0 {
		(((1u64 << fraction_bits) | fraction) << (23 - fraction_bits), exponent - layout.bias)
	} else {
		let leading = 63 - fraction.leading_zeros() as i32;
		(fraction << (23 - leading) as u32, leading + 1 - layout.bias - fraction_bits)
	};

	if exponent >= -126 {
		return sign | (((exponent + 127) as u32) << 23) | (significand as u32 & 0x007f_ffff);
	}

	// Below F32's smallest normal there is no exponent field left to hold the
	// difference, so it goes in the fraction instead. This is not a corner case:
	// `BF16`'s subnormals run from 2^-126 down to 2^-133 and every one of them
	// lands inside F32's subnormal range, and reading them as normals would hand
	// them an exponent of about -7.
	let shift = -exponent - 126;

	if shift >= 64 {
		return sign;
	}

	sign | ((significand >> shift) as u32 & 0x007f_ffff)
}

/// The packed word of `layout` an `F32` narrows to, rounded to nearest, ties
/// to even.
fn from_f32_bits(bits: u32, layout: Layout) -> u64 {
	let sign = ((bits as u64) & 0x8000_0000) >> (31 - layout.exponent_bits - layout.fraction_bits);
	let magnitude = (bits as u64) & 0x7fff_ffff;

	if magnitude == 0 {
		return sign;
	}

	// Non-finite values have to be respelled, because no two formats spell them
	// the same way. NaN is the one pattern `Layout::nan_word` names, which is the
	// only NaN `E4M3` can hold and a saturated exponent with a full fraction
	// elsewhere. Infinity is the other: where the format has one it is the
	// saturated exponent alone, and where it does not an infinity is out of
	// range like any other value and saturates to the largest finite word.
	if magnitude >= 0x7f80_0000 {
		return sign | if magnitude > 0x7f80_0000 { layout.nan_word() } else { layout.overflow_word(false) };
	}

	// An F32 subnormal has no implicit one to hang the value off, so the fraction
	// is read as `0.f` rather than as `1.f`. It is then normalised — shifted up
	// until its leading bit sits where a normal significand keeps it, with the
	// exponent taking the difference — because the shift below counts on the
	// leading bit being in that place. Left unnormalised it would come out of a
	// trip 2^23 too large: `BF16::from_bits(0x0001)` would widen to 2^-133 and
	// narrow straight back to 2^-126.
	let exponent_field = ((magnitude >> 23) & 0xff) as i32;
	let (significand, unbiased) = if exponent_field == 0 {
		let fraction = magnitude & 0x007f_ffff;
		let leading = 63 - fraction.leading_zeros() as i32;
		(fraction << (23 - leading) as u32, leading - 149)
	} else {
		((magnitude & 0x007f_ffff) | (1u64 << 23), exponent_field - 127)
	};

	let fraction_bits = layout.fraction_bits as i32;
	let biased = unbiased + layout.bias;

	// Below the smallest normal here the exponent field has nothing to say, so
	// the fraction absorbs what the exponent would have carried. The shift grows
	// as the value falls, which is exactly what makes a tiny value flush to zero
	// instead of wrapping.
	let (shift, normal) = if biased >= 1 {
		(23 - fraction_bits, true)
	} else {
		(24 - unbiased - layout.bias - fraction_bits, false)
	};

	let rounded = round_shift_right(significand, shift);

	if !normal {
		// `rounded` can reach 2^fraction_bits only by rounding up to the
		// smallest normal, and that pattern is its own encoding: the bit lands
		// in the exponent field where a normal would put it.
		return sign | rounded;
	}

	// Rounding can carry out of the fraction and into the exponent, which is the
	// difference between 65504 and 65536 in `F16`.
	let (mantissa, field) = if rounded >> (fraction_bits + 1) != 0 {
		(rounded >> 1, biased as u64 + 1)
	} else {
		(rounded, biased as u64)
	};

	if field >= layout.max_exponent() {
		// Past the top of the range. `E4M3` has nowhere to put an infinity, so it
		// holds at its largest value instead, and every other format holds there
		// too.
		return sign | if layout.has_infinity { layout.exponent_mask() } else { layout.max_finite };
	}

	sign | (field << layout.fraction_bits) | (mantissa & layout.fraction_mask())
}

/// The `F32` pattern a packed word of `layout` widens to, exposed so the tests
/// can walk a format's whole word space.
pub(crate) fn to_f32_bits_for_test(word: u64, layout: Layout) -> u32 {
	to_f32_bits(word, layout)
}

/// Widen `value` to `F32`, exactly, for any format at or below 32 bits.
pub(crate) fn to_f32<T: Float>(value: T) -> F32 {
	let layout = T::layout();
	F32::from_bits(to_f32_bits(value.to_f64_word(), layout))
}

/// Narrow `value` to `T`, rounding to nearest with ties to even.
pub(crate) fn from_f32<T: Float>(value: F32) -> T {
	let layout = T::layout();
	T::from_f64_word(from_f32_bits(value.to_bits(), layout))
}

/// Widen `value` to `F64`, exactly.
pub(crate) fn to_f64<T: Float>(value: T) -> F64 {
	let as_f32 = to_f32::<T>(value);

	if as_f32.is_nan() {
		return F64::from_native(f64::NAN);
	}

	F64::from_native(as_f32.to_native() as f64)
}

/// Narrow `value` from `F64` to `T`, rounding to nearest with ties to even.
pub(crate) fn from_f64<T: Float>(value: F64) -> T {
	from_f32::<T>(F32::from_native(value.to_native() as f32))
}

/// Convert between any two formats at or below 32 bits, through `F32`.
///
/// The trip is a widening or a narrowing depending on which side holds the
/// fewer fraction bits, and the lossy direction rounds once, by the same rule a
/// compiler applies when it says a float literal changed.
///
/// `F64` is not one of those formats and cannot be reached through this: it has
/// 53 fraction bits, so routing it here would round it to 24 before the real
/// target ever saw it. Use [`to_f64`] and [`from_f64`] for it. Those go via
/// `F32` too, which is a double rounding, and it is worth naming rather than
/// hiding: `F64` to `F16` can land a step away from the single-rounding answer.
/// Nothing in this crate stores `F64` in a narrow format, so the case does not
/// arise here.
pub(crate) fn convert<T, U>(value: T) -> U
where
	T: Float,
	U: Float,
{
	from_f32::<U>(to_f32::<T>(value))
}

/// The byte width of a format, from the registry rather than from the type.
///
/// A tag read off a buffer has to be turned into a width before anything can be
/// carved, and asking the registry keeps that from drifting from what the types
/// actually are.
pub(crate) fn byte_size_of(dtype: DType) -> usize {
	dtype.byte_size()
}

/// The encoding of a format, from the registry.
pub(crate) fn layout_of(dtype: DType) -> Layout {
	dtype.layout()
}

/// The byte width and the encoding fields the registry carries for `dtype`.
///
/// Layout questions belong to the format, not the element type, so a facade
/// that meets a buffer by its tag asks the registry through this.
pub(crate) fn dtype_info(dtype: DType) -> (usize, u32, u32, i32) {
	let layout = layout_of(dtype);
	(
		byte_size_of(dtype),
		layout.exponent_bits,
		layout.fraction_bits,
		layout.bias,
	)
}

/// Run `round`, `recip` and `tanh` on the value `word` names in `dtype`, for
/// every format the registry knows, and widen the results.
///
/// This is how the narrow formats get exercised: their arithmetic is real
/// arithmetic on the actual values, every format read and written through its
/// own `Float` machinery rather than any reinterpretation of its storage. The
/// tuple carries, in order, the rounded word, the reciprocal word, the tanh
/// word, the bits of the value widened to `F64`, and the `F32` pattern the raw
/// word widens to.
pub(crate) fn dtype_scan(dtype: DType, word: u64) -> (u64, u64, u64, u64, u64) {
	match dtype {
		DType::F32 => narrow_scan::<F32>(word),
		DType::F16 => narrow_scan::<F16>(word),
		DType::BF16 => narrow_scan::<BF16>(word),
		DType::TF32 => narrow_scan::<TF32>(word),
		DType::E4M3 => narrow_scan::<E4M3>(word),
		DType::E5M2 => narrow_scan::<E5M2>(word),
		DType::F64 => {
			let (rounded, recip, tanh) = scan_math::<F64>(word);
			// `F64` is the widening target itself, so its widened form is the
			// word as read, and it has no `F32` pattern of interest.
			(rounded, recip, tanh, word, 0)
		}
	}
}

/// The scan for one of the formats at or below 32 bits: the three operations,
/// plus the `F64` and `F32` widening of the original value.
fn narrow_scan<U: Float>(word: u64) -> (u64, u64, u64, u64, u64) {
	let value = U::from_f64_word(word);
	let (rounded, recip, tanh) = scan_math::<U>(word);
	let widened = to_f64::<U>(value).to_bits();
	let pattern = to_f32_bits_for_test(value.to_f64_word(), U::layout()) as u64;
	(rounded, recip, tanh, widened, pattern)
}

/// The three operations of a scan, as packed words of the format.
fn scan_math<U: Float>(word: u64) -> (u64, u64, u64) {
	let value = U::from_f64_word(word);
	let rounded = value.round().to_f64_word();
	let recip = value.recip().to_f64_word();
	let tanh = value.tanh().to_f64_word();
	(rounded, recip, tanh)
}

/// Convert every element of `source` into a buffer of `target`.
///
/// The two buffers must hold the same number of elements: this is a change of
/// format, not of shape, and a length mismatch is a mistake the caller should
/// hear about rather than something to truncate around.
pub(crate) fn convert_buffer<T: Float, U: Float>(
	source: &Buf<'_, T>,
	target: &mut Buf<'_, U>,
) -> Result<usize, MathError> {
	let count = T::carve_count(source)?;

	if target.len() < count {
		return Err(CarveError::SizeMismatch.into());
	}

	for index in 0..count {
		let value = T::carve_read(source, index)?;
		U::carve_write(target, index, convert(value))?;
	}

	Ok(count)
}

#[inline]
pub(crate) fn ldexpd(x: f64, exp: i32) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000_0000_0000u64;
    let mant = bits & 0x000f_ffff_ffff_ffffu64;
    let mut e = ((bits >> 52) & 0x7ff) as i32 - 1023;
    e += exp;
    if e <= -1023 {
        return 0.0;
    }
    if e >= 1024 {
        return f64::INFINITY;
    }
    let new_bits = sign | (((e + 1023) as u64) << 52) | mant;
    f64::from_bits(new_bits)
}

// ---------------------------------------------------------------------------
// The math layer
// ---------------------------------------------------------------------------

/// The math layer, the second tier of the tower.
///
/// It is stateless by design: every operation reads and writes the buffers it is
/// handed, and the only thing it carries is the element type. [`LmlpBase`] owns
/// one as a field and hands it carves of its scratch, so arithmetic never
/// allocates and never names a backend of its own.
///
/// [`LmlpBase`]: crate::base::LmlpBase
pub(crate) struct LmlpMath<T> {
    _marker: PhantomData<T>,
}

impl<T: Float> LmlpMath<T> {
    pub(crate) const fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }

    pub(crate) fn abs(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        abs_carve(buffer)
    }

    pub(crate) fn recip(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        reciprocal_carve(buffer)
    }

    pub(crate) fn ldexp(
        &self,
        buffer: &mut Buf<'_, T>,
        exponent: i32,
    ) -> Result<usize, MathError> {
        ldexp_carve(buffer, exponent)
    }

    pub(crate) fn exp(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::exp_log::exp_carve(buffer)
    }

    pub(crate) fn ln(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::exp_log::ln_carve(buffer)
    }

    pub(crate) fn sqrt(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::roots::sqrt_carve(buffer)
    }

    pub(crate) fn sin(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::trig::sin_carve(buffer)
    }

    pub(crate) fn cos(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::trig::cos_carve(buffer)
    }

    pub(crate) fn tanh(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::trig::tanh_carve(buffer)
    }

    pub(crate) fn relu(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::logistic::relu_carve(buffer)
    }

    pub(crate) fn sigmoid(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::logistic::sigmoid_carve(buffer)
    }

    pub(crate) fn round(&self, buffer: &mut Buf<'_, T>) -> Result<usize, MathError> {
        super::rounding::round_carve(buffer)
    }

    pub(crate) fn pow(
        &self,
        buffer: &mut Buf<'_, T>,
        exponent: T,
    ) -> Result<usize, MathError> {
        super::power::pow_carve(buffer, exponent)
    }

    pub(crate) fn convert_into<U: Float>(
        &self,
        source: &Buf<'_, T>,
        target: &mut Buf<'_, U>,
    ) -> Result<usize, MathError> {
        convert_buffer(source, target)
    }

    pub(crate) fn floor_f32(&self, buffer: &mut Buf<'_, F32>) -> Result<usize, MathError> {
        floorf_carve(buffer)
    }

    pub(crate) fn floor_f64(&self, buffer: &mut Buf<'_, F64>) -> Result<usize, MathError> {
        floord_carve(buffer)
    }

    pub(crate) fn floor_f16(&self, buffer: &mut Buf<'_, F16>) -> Result<usize, MathError> {
        floorf16_carve(buffer)
    }

    pub(crate) fn floor_bf16(&self, buffer: &mut Buf<'_, BF16>) -> Result<usize, MathError> {
        floor_bf16_carve(buffer)
    }

    pub(crate) fn recip_into(
        &self,
        source: &Buf<'_, T>,
        target: &mut Buf<'_, T>,
    ) -> Result<usize, MathError> {
        elementwise::unary_into(source, target, <T as Float>::recip)
    }

    pub(crate) fn binary_add(
        &self,
        left: &mut Buf<'_, T>,
        right: &Buf<'_, T>,
    ) -> Result<usize, MathError> {
        elementwise::binary(left, right, core::ops::Add::add)
    }

    pub(crate) fn clamp(
        &self,
        buffer: &mut Buf<'_, T>,
        low: T,
        high: T,
    ) -> Result<usize, MathError> {
        elementwise::clamped(buffer, low, high)
    }
}
