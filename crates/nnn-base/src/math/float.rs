use super::exp_log::{expd, expf, lnd, lnf};
use super::logistic::{sigmoidd, sigmoidf};
use super::power::{powd, powf};
use super::roots::{sqrtd, sqrtf};
use super::rounding::{roundd, roundf};
use super::trig::{cosd, cosf, sind, sinf, tanhd, tanhf};
use crate::scratch::carve::{Carve, CarveError};
use crate::scratch::typed::Buf;

/// Where a format keeps its exponent and its fraction, and what it does at the
/// ends of its range.
///
/// Every format in this crate is `sign | exponent | fraction`, so the three
/// widths plus the bias are the whole description. `has_infinity` and
/// `max_finite` are what separate `E4M3` from the rest: it spends its top
/// exponent on ordinary numbers, so an overflow has to land on the largest
/// finite value instead of an infinity that the format cannot spell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
	pub(crate) exponent_bits: u32,
	pub(crate) fraction_bits: u32,
	pub(crate) bias: i32,
	pub(crate) has_infinity: bool,
	/// The largest finite magnitude, as a bit pattern. For a format with an
	/// infinity this is the one below it, so the two differ by one step.
	pub(crate) max_finite: u64,
	/// Bits at the bottom of the stored word that carry no value.
	///
	/// Most formats fill their word exactly and this is zero. `TF32` is the
	/// exception: it is an `F32` with 13 mantissa bits dropped, so its 19 live
	/// bits sit at the top of a 32-bit word and a checkpoint that stores it
	/// bytes-for-byte is readable as an `F32`. Every mask below is stated on the
	/// packed form, and [`Layout::pack`] and [`Layout::unpack`] are the only
	/// places the two differ.
	pub(crate) storage_shift: u32,
}

impl Layout {
	pub(crate) const fn new(
		exponent_bits: u32,
		fraction_bits: u32,
		bias: i32,
		has_infinity: bool,
		max_finite: u64,
	) -> Self {
		Self { exponent_bits, fraction_bits, bias, has_infinity, max_finite, storage_shift: 0 }
	}

	pub(crate) const fn shifted(
		exponent_bits: u32,
		fraction_bits: u32,
		bias: i32,
		has_infinity: bool,
		max_finite: u64,
		storage_shift: u32,
	) -> Self {
		Self { exponent_bits, fraction_bits, bias, has_infinity, max_finite, storage_shift }
	}

	/// The stored word down to the format's own bits.
	pub(crate) const fn pack(self, storage: u64) -> u64 {
		storage >> self.storage_shift
	}

	/// The format's own bits back up to where a stored word holds them.
	pub(crate) const fn unpack(self, packed: u64) -> u64 {
		packed << self.storage_shift
	}

	/// The sign bit where the storage actually keeps it.
	pub(crate) const fn storage_sign_mask(self) -> u64 {
		1u64 << (self.exponent_bits + self.fraction_bits + self.storage_shift)
	}

	/// The single bit holding the sign.
	pub(crate) const fn sign_mask(self) -> u64 {
		1u64 << (self.exponent_bits + self.fraction_bits)
	}

	/// Every exponent bit, already shifted above the fraction.
	pub(crate) const fn exponent_mask(self) -> u64 {
		((1u64 << self.exponent_bits) - 1) << self.fraction_bits
	}

	pub(crate) const fn fraction_mask(self) -> u64 {
		(1u64 << self.fraction_bits) - 1
	}

	/// The exponent field of infinity, or of the top normal where there is none.
	pub(crate) const fn max_exponent(self) -> u64 {
		(1u64 << self.exponent_bits) - 1
	}

	pub(crate) const fn is_negative(self, word: u64) -> bool {
		word & self.sign_mask() != 0
	}

	/// A saturated exponent means infinity or NaN, but only where the format has
	/// an infinity to spend. `E4M3` uses the same field for real numbers, so the
	/// check has to be told the difference.
	pub(crate) const fn exponent_saturated(self, word: u64) -> bool {
		self.has_infinity && (word & self.exponent_mask()) == self.exponent_mask()
	}

	/// Whether a pattern is NaN.
	///
	/// Where the format has an infinity, a saturated exponent plus any fraction
	/// at all is NaN — the fraction is the payload, and there are many of them.
	/// `E4M3` has no infinity, so its top pattern is the only non-finite word it
	/// can spell, and every other pattern below it is a real number.
	pub(crate) const fn is_nan_word(self, word: u64) -> bool {
		if self.has_infinity {
			(word & self.exponent_mask()) == self.exponent_mask()
				&& (word & self.fraction_mask()) != 0
		} else {
			word & self.magnitude_mask() == self.nan_word()
		}
	}

	/// Whether a pattern is an ordinary number.
	///
	/// This is not the negation of [`Layout::is_nan_word`]: infinity is finite
	/// in neither sense but is not NaN, so the two questions are asked apart.
	pub(crate) const fn is_finite_word(self, word: u64) -> bool {
		if self.has_infinity {
			(word & self.exponent_mask()) != self.exponent_mask()
		} else {
			word & self.magnitude_mask() != self.nan_word()
		}
	}

	pub(crate) const fn magnitude_mask(self) -> u64 {
		!self.sign_mask()
	}

	/// The one non-finite pattern a format without an infinity can carry.
	pub(crate) const fn nan_word(self) -> u64 {
		if self.has_infinity { self.exponent_mask() | self.fraction_mask() }
		else { self.max_finite + 1 }
	}

	/// What an out-of-range result becomes: an infinity where one exists, the
	/// largest finite value where it does not.
	pub(crate) const fn overflow_word(self, negative: bool) -> u64 {
		let magnitude = if self.has_infinity { self.exponent_mask() } else { self.max_finite };
		if negative { magnitude | self.sign_mask() } else { magnitude }
	}

	pub const fn infinity_word(self, negative: bool) -> u64 {
		self.overflow_word(negative)
	}
}

/// The formats this crate knows, in `DTYPE_TAG` order.
///
/// The tags are what a serialized buffer carries, so `F32` through `BF16` keep
/// the numbers they already had and the new ones are appended. A tag is the only
/// thing that has to survive a round trip through a file, which is why the enum
/// exists rather than each type carrying its own number alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
	F32 = 0,
	F64 = 1,
	F16 = 2,
	BF16 = 3,
	TF32 = 4,
	E4M3 = 5,
	E5M2 = 6,
}

impl DType {
	pub const ALL: [DType; 7] = [
		DType::F32,
		DType::F64,
		DType::F16,
		DType::BF16,
		DType::TF32,
		DType::E4M3,
		DType::E5M2,
	];

	pub const fn tag(self) -> u8 {
		self as u8
	}

	/// The format for a tag read off a buffer, or `None` when the tag names a
	/// format this build does not have.
	pub const fn from_tag(tag: u8) -> Option<DType> {
		match tag {
			0 => Some(DType::F32),
			1 => Some(DType::F64),
			2 => Some(DType::F16),
			3 => Some(DType::BF16),
			4 => Some(DType::TF32),
			5 => Some(DType::E4M3),
			6 => Some(DType::E5M2),
			_ => None,
		}
	}

	/// The width of one element. Zero for `F64`, whose 64 bits do not fit the
	/// narrow path this crate does conversions through.
	pub(crate) const fn byte_size(self) -> usize {
		match self {
			DType::F32 | DType::TF32 => 4,
			DType::F64 => 8,
			DType::F16 | DType::BF16 => 2,
			DType::E4M3 | DType::E5M2 => 1,
		}
	}

	pub(crate) const fn layout(self) -> Layout {
		match self {
			DType::F32 => Layout::new(8, 23, 127, true, 0x7f_7f_ff_ff),
			DType::F64 => Layout::new(11, 52, 1023, true, 0x7f_ef_ff_ff_ff_ff_ff),
			DType::F16 => Layout::new(5, 10, 15, true, 0x7b_ff),
			DType::BF16 => Layout::new(8, 7, 127, true, 0x7f_7f),
			// The 19 live bits sit at the top of the word, so the storage carries
			// 13 empty bits below the fraction. `max_finite` is on the packed form
			// like every other mask here: exponent 254 with all ten fraction bits
			// set is 0x3F_BFF, and stored that is 0x7F_7F_E0_00.
			DType::TF32 => Layout::shifted(8, 10, 127, true, 0x00_3f_bf_ff, 13),
			DType::E4M3 => Layout::new(4, 3, 7, false, 0x7e),
			DType::E5M2 => Layout::new(5, 2, 15, true, 0x7b),
		}
	}
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32(u32);

impl F32 {
	pub(crate) const fn from_bits(bits: u32) -> Self {
		Self(bits)
	}

	pub const fn to_bits(self) -> u32 {
		self.0
	}

	pub(crate) const fn to_native(self) -> f32 {
		f32::from_bits(self.0)
	}

	pub(crate) const fn from_native(value: f32) -> Self {
		Self(value.to_bits())
	}
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F64(u64);

impl F64 {
	pub(crate) const fn from_bits(bits: u64) -> Self {
		Self(bits)
	}

	pub const fn to_bits(self) -> u64 {
		self.0
	}

	pub(crate) const fn to_native(self) -> f64 {
		f64::from_bits(self.0)
	}

	pub(crate) const fn from_native(value: f64) -> Self {
		Self(value.to_bits())
	}
}

pub trait Float:
    Copy
    + PartialOrd
    + Send
    + Sync
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
    + core::ops::Neg<Output = Self>
    + core::ops::AddAssign
    + core::ops::SubAssign
    + core::ops::MulAssign
    + core::ops::DivAssign
{
    const ZERO: Self;
    const ONE: Self;
    const HALF: Self;
    const TWO: Self;
    const NEG_INF: Self;
    const POS_INF: Self;
    const EPSILON: Self;

    fn abs(self) -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tanh(self) -> Self;
    fn relu(self) -> Self;
    fn sigmoid(self) -> Self;
    fn powf(self, other: Self) -> Self;
    fn round(self) -> Self;
    fn recip(self) -> Self;

    const BYTE_SIZE: usize;
    const DTYPE_TAG: u8;
    fn layout() -> Layout;
    fn dtype() -> DType;
    /// The value's encoding as a 64-bit pattern, widened from whatever the
    /// format actually stores.
    ///
    /// Conversions run through this rather than through the native type, so one
    /// piece of code serves every format and `F64` is the only one that cannot
    /// go through it — its 64 bits do not fit.
    fn to_f64_word(self) -> u64;
    /// The format's own encoding of a 64-bit pattern, narrowing to whatever it
    /// stores. The pattern is expected to already be inside this format's range.
    fn from_f64_word(word: u64) -> Self;
    fn write_le_bytes(self, out: &mut [u8]);
    fn from_le_bytes(bytes: &[u8]) -> Self;

	fn carve_count(buffer: &Buf<'_, Self>) -> Result<usize, CarveError> {
		if buffer.is_empty() {
			return Err(CarveError::SizeMismatch);
		}

		Ok(buffer.len())
	}

	fn carve_fill(
		buffer: &mut Buf<'_, Self>,
		value: Self,
	) -> Result<usize, CarveError> {
		let count = Self::carve_count(buffer)?;

		for slot in buffer.as_mut_slice() {
			*slot = value;
		}

		Ok(count)
	}

	fn carve_map(
		buffer: &mut Buf<'_, Self>,
		op: impl Fn(Self) -> Self,
	) -> Result<usize, CarveError> {
		let count = Self::carve_count(buffer)?;

		for slot in buffer.as_mut_slice() {
			*slot = op(*slot);
		}

		Ok(count)
	}

	fn carve_read(
		buffer: &Buf<'_, Self>,
		index: usize,
	) -> Result<Self, CarveError> {
		if index >= buffer.len() {
			return Err(CarveError::OutOfMemory);
		}

		Ok(buffer.as_slice()[index])
	}

	fn carve_write(
		buffer: &mut Buf<'_, Self>,
		index: usize,
		value: Self,
	) -> Result<(), CarveError> {
		if index >= buffer.len() {
			return Err(CarveError::OutOfMemory);
		}

		buffer.as_mut_slice()[index] = value;

		Ok(())
	}

	fn carve_dot(
		left: &Buf<'_, Self>,
		right: &Buf<'_, Self>,
	) -> Result<Self, CarveError> {
		if left.len() != right.len() {
			return Err(CarveError::SizeMismatch);
		}

		let mut total = Self::ZERO;

		for (left_value, right_value) in left.as_slice().iter().zip(right.as_slice()) {
			total += *left_value * *right_value;
		}

		Ok(total)
	}

	fn carve_sum(buffer: &Buf<'_, Self>) -> Result<Self, CarveError> {
		let mut total = Self::ZERO;

		for value in buffer.as_slice() {
			total += *value;
		}

		Ok(total)
	}

	fn carve_write_le_bytes(
		&self,
		bytes: &mut Buf<'_, u8>,
		index: usize,
	) -> Result<(), CarveError> {
		let start = match index.checked_mul(Self::BYTE_SIZE) {
			Some(start) => start,
			None => return Err(CarveError::SizeMismatch),
		};

		let end = match start.checked_add(Self::BYTE_SIZE) {
			Some(end) => end,
			None => return Err(CarveError::SizeMismatch),
		};

		if end > bytes.len() {
			return Err(CarveError::SizeMismatch);
		}

		self.write_le_bytes(&mut bytes.as_mut_slice()[start..end]);

		Ok(())
	}

	fn carve_from_le_bytes(
		bytes: &Buf<'_, u8>,
		index: usize,
	) -> Result<Self, CarveError> {
		let start = match index.checked_mul(Self::BYTE_SIZE) {
			Some(start) => start,
			None => return Err(CarveError::SizeMismatch),
		};

		let end = match start.checked_add(Self::BYTE_SIZE) {
			Some(end) => end,
			None => return Err(CarveError::SizeMismatch),
		};

		if end > bytes.len() {
			return Err(CarveError::SizeMismatch);
		}

		Ok(Self::from_le_bytes(&bytes.as_slice()[start..end]))
	}
}

impl Float for F32 {
    const ZERO: Self = Self::from_bits(0x0000_0000);
    const ONE: Self = Self::from_bits(0x3f80_0000);
    const HALF: Self = Self::from_bits(0x3f00_0000);
    const TWO: Self = Self::from_bits(0x4000_0000);
    const NEG_INF: Self = Self::from_bits(0xff80_0000);
    const POS_INF: Self = Self::from_bits(0x7f80_0000);
    const EPSILON: Self = Self::from_bits(0x3400_0000);

    #[inline]
    fn abs(self) -> Self {
        Self::from_bits(self.0 & 0x7fff_ffff)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self.to_native())
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self.to_native())
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::from_native(sqrtf(self.to_native()))
    }
    #[inline]
    fn exp(self) -> Self {
        Self::from_native(expf(self.to_native()))
    }
    #[inline]
    fn ln(self) -> Self {
        Self::from_native(lnf(self.to_native()))
    }
    #[inline]
    fn sin(self) -> Self {
        Self::from_native(sinf(self.to_native()))
    }
    #[inline]
    fn cos(self) -> Self {
        Self::from_native(cosf(self.to_native()))
    }
    #[inline]
    fn tanh(self) -> Self {
        Self::from_native(tanhf(self.to_native()))
    }
    #[inline]
    fn relu(self) -> Self {
        self.max(Self::ZERO)
    }
    #[inline]
    fn sigmoid(self) -> Self {
        Self::from_native(sigmoidf(self.to_native()))
    }
    #[inline]
    fn powf(self, other: Self) -> Self {
        Self::from_native(powf(self.to_native(), other.to_native()))
    }
    #[inline]
    fn round(self) -> Self {
        Self::from_native(roundf(self.to_native()))
    }
    #[inline]
    fn recip(self) -> Self {
        Self::from_native(1.0 / self.to_native())
    }

    const BYTE_SIZE: usize = 4;
    const DTYPE_TAG: u8 = 0;
    #[inline]
    fn layout() -> Layout {
        DType::F32.layout()
    }
    #[inline]
    fn dtype() -> DType {
        DType::F32
    }
    #[inline]
    fn to_f64_word(self) -> u64 {
        self.0 as u64
    }
    #[inline]
    fn from_f64_word(word: u64) -> Self {
        Self::from_bits(word as u32)
    }
    #[inline]
    fn write_le_bytes(self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.0.to_le_bytes());
    }
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_bits(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }
}

// The arithmetic goes through the native type on purpose: these two are the
// formats the hardware already has, and a newtype is only a wrapper if the
// operations underneath stay the same ones.
impl core::ops::Add for F32 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() + rhs.to_native())
    }
}

impl core::ops::Sub for F32 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() - rhs.to_native())
    }
}

impl core::ops::Mul for F32 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() * rhs.to_native())
    }
}

impl core::ops::Div for F32 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() / rhs.to_native())
    }
}

impl core::ops::Neg for F32 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_native(-self.to_native())
    }
}

impl core::ops::AddAssign for F32 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for F32 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for F32 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for F32 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl PartialEq for F32 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(core::cmp::Ordering::Equal)
    }
}

impl PartialOrd for F32 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_native().partial_cmp(&other.to_native())
    }
}

impl Float for F64 {
    const ZERO: Self = Self::from_bits(0x0000_0000_0000_0000);
    const ONE: Self = Self::from_bits(0x3ff0_0000_0000_0000);
    const HALF: Self = Self::from_bits(0x3fe0_0000_0000_0000);
    const TWO: Self = Self::from_bits(0x4000_0000_0000_0000);
    const NEG_INF: Self = Self::from_bits(0xfff0_0000_0000_0000);
    const POS_INF: Self = Self::from_bits(0x7ff0_0000_0000_0000);
    const EPSILON: Self = Self::from_bits(0x3cb0_0000_0000_0000);

    #[inline]
    fn abs(self) -> Self {
        Self::from_bits(self.0 & 0x7fff_ffff_ffff_ffff)
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self.to_native())
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self.to_native())
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::from_native(sqrtd(self.to_native()))
    }
    #[inline]
    fn exp(self) -> Self {
        Self::from_native(expd(self.to_native()))
    }
    #[inline]
    fn ln(self) -> Self {
        Self::from_native(lnd(self.to_native()))
    }
    #[inline]
    fn sin(self) -> Self {
        Self::from_native(sind(self.to_native()))
    }
    #[inline]
    fn cos(self) -> Self {
        Self::from_native(cosd(self.to_native()))
    }
    #[inline]
    fn tanh(self) -> Self {
        Self::from_native(tanhd(self.to_native()))
    }
    #[inline]
    fn relu(self) -> Self {
        self.max(Self::ZERO)
    }
    #[inline]
    fn sigmoid(self) -> Self {
        Self::from_native(sigmoidd(self.to_native()))
    }
    #[inline]
    fn powf(self, other: Self) -> Self {
        Self::from_native(powd(self.to_native(), other.to_native()))
    }
    #[inline]
    fn round(self) -> Self {
        Self::from_native(roundd(self.to_native()))
    }
    #[inline]
    fn recip(self) -> Self {
        Self::from_native(1.0 / self.to_native())
    }

    const BYTE_SIZE: usize = 8;
    const DTYPE_TAG: u8 = 1;
    #[inline]
    fn layout() -> Layout {
        DType::F64.layout()
    }
    #[inline]
    fn dtype() -> DType {
        DType::F64
    }
    #[inline]
    fn to_f64_word(self) -> u64 {
        self.0
    }
    #[inline]
    fn from_f64_word(word: u64) -> Self {
        Self::from_bits(word)
    }
    #[inline]
    fn write_le_bytes(self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.0.to_le_bytes());
    }
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        Self::from_bits(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }
}

impl core::ops::Add for F64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() + rhs.to_native())
    }
}

impl core::ops::Sub for F64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() - rhs.to_native())
    }
}

impl core::ops::Mul for F64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() * rhs.to_native())
    }
}

impl core::ops::Div for F64 {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self::from_native(self.to_native() / rhs.to_native())
    }
}

impl core::ops::Neg for F64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::from_native(-self.to_native())
    }
}

impl core::ops::AddAssign for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::ops::SubAssign for F64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl core::ops::MulAssign for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl core::ops::DivAssign for F64 {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl PartialEq for F64 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(core::cmp::Ordering::Equal)
    }
}

impl PartialOrd for F64 {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.to_native().partial_cmp(&other.to_native())
    }
}

// PartialEq and PartialOrd are derived below the structs instead, in
// `impl_float16!`, because the bit pattern is not the value: a derived
// ordering would read 0x8000 (-0.0) as larger than 0x0000 (+0.0), and would
// call two zeros different. Every comparison a caller makes — `sqrt` on a
// negative, `ln` on -0.0, a reLU threshold — goes through here, so getting the
// order right is not a detail.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F16(u16);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct BF16(u16);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct TF32(u32);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct E4M3(u8);

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct E5M2(u8);

impl F16 {
    pub(crate) const fn from_bits(bits: u16) -> Self { Self(bits) }
    pub const fn to_bits(self) -> u16 { self.0 }
}

impl BF16 {
    pub(crate) const fn from_bits(bits: u16) -> Self { Self(bits) }
    pub const fn to_bits(self) -> u16 { self.0 }
}

impl TF32 {
    pub(crate) const fn from_bits(bits: u32) -> Self { Self(bits) }
    pub const fn to_bits(self) -> u32 { self.0 }
}

impl E4M3 {
    pub(crate) const fn from_bits(bits: u8) -> Self { Self(bits) }
    pub const fn to_bits(self) -> u8 { self.0 }
}

impl E5M2 {
    pub(crate) const fn from_bits(bits: u8) -> Self { Self(bits) }
    pub const fn to_bits(self) -> u8 { self.0 }
}

/// Flips the sign bit of a packed word.
///
/// Subtraction is addition with a negated right-hand side, so this is the one
/// bit operation `Sub` needs. It lives outside the operator impl on purpose:
/// a bare `^` inside `sub` reads as a mistake to both readers and lints.
#[inline]
fn flip_sign(word: u64, sign_mask: u64) -> u64 {
    word ^ sign_mask
}

/// The three arithmetic kernels, generic over every format at or below 32 bits.
///
/// The words arrive as `u64` whatever the format actually stores, and every mask
/// comes from the [`Layout`] rather than from the width. That is what lets the
/// same code serve an eight-bit format and `TF32`: the differences that used to
/// be baked into `u16` are now the only things a caller has to supply.
fn fp_add_bits(left: u64, right: u64, layout: Layout) -> u64 {
    let Layout { fraction_bits, bias, has_infinity, .. } = layout;
    let sign_mask = layout.sign_mask();
    let exponent_mask = layout.exponent_mask();
    let fraction_mask = layout.fraction_mask();
    let max_exponent = layout.max_exponent() as i32;
    let mut left_exp = ((left & exponent_mask) >> fraction_bits) as i32;
    let mut right_exp = ((right & exponent_mask) >> fraction_bits) as i32;
    let left_sign = layout.is_negative(left);
    let right_sign = layout.is_negative(right);
    let mut left_mantissa = (left & fraction_mask) as i64;
    let mut right_mantissa = (right & fraction_mask) as i64;

    // NaN first, in every format. A saturated exponent only means infinity where
    // the format has one; `E4M3` reaches this point holding ordinary numbers.
    if layout.is_nan_word(left) {
        return left;
    }

    if layout.is_nan_word(right) {
        return right;
    }

    if has_infinity && (left_exp == max_exponent || right_exp == max_exponent) {
        return if left_exp == max_exponent { left } else { right };
    }

    if left_exp == 0 { left_exp = 1; } else { left_mantissa |= 1i64 << fraction_bits; }
    if right_exp == 0 { right_exp = 1; } else { right_mantissa |= 1i64 << fraction_bits; }
    let left_unbiased_exp = left_exp - bias;
    let right_unbiased_exp = right_exp - bias;
    let common_exp = left_unbiased_exp.max(right_unbiased_exp);
    left_mantissa <<= 3;
    right_mantissa <<= 3;
    if common_exp > left_unbiased_exp { left_mantissa >>= (common_exp - left_unbiased_exp).min(62) as u32; }
    if common_exp > right_unbiased_exp { right_mantissa >>= (common_exp - right_unbiased_exp).min(62) as u32; }
    let (sign, mut mantissa) = if left_sign == right_sign {
        (left_sign, left_mantissa + right_mantissa)
    } else if left_mantissa >= right_mantissa {
        (left_sign, left_mantissa - right_mantissa)
    } else {
        (right_sign, right_mantissa - left_mantissa)
    };
    if mantissa == 0 { return 0; }
    let target = fraction_bits + 3;
    let mut exponent = common_exp + bias;
    while mantissa >= (1i64 << (target + 1)) { mantissa = (mantissa + 1) >> 1; exponent += 1; }
    while mantissa < (1i64 << target) && exponent > 1 { mantissa <<= 1; exponent -= 1; }
    if exponent >= max_exponent { return layout.overflow_word(sign); }
    let fraction = if exponent == 1 && mantissa < (1i64 << fraction_bits) {
        mantissa >> 3
    } else {
        (mantissa >> 3) & fraction_mask as i64
    } as u64;
    (if sign { sign_mask } else { 0 }) | ((exponent as u64) << fraction_bits) | fraction
}

fn fp_mul_bits(left: u64, right: u64, layout: Layout) -> u64 {
    let Layout { fraction_bits, bias, .. } = layout;
    let sign_mask = layout.sign_mask();
    let exponent_mask = layout.exponent_mask();
    let fraction_mask = layout.fraction_mask();
    let max_exponent = layout.max_exponent();
    let negative = layout.is_negative(left) != layout.is_negative(right);
    let left_exp = (left & exponent_mask) >> fraction_bits;
    let right_exp = (right & exponent_mask) >> fraction_bits;
    let left_fraction = left & fraction_mask;
    let right_fraction = right & fraction_mask;
    if layout.is_nan_word(left) || layout.is_nan_word(right) {
        return layout.nan_word();
    }
    if layout.has_infinity && (left_exp == max_exponent || right_exp == max_exponent) {
        return layout.overflow_word(negative);
    }
    if (left_exp == 0 && left_fraction == 0) || (right_exp == 0 && right_fraction == 0) { return if negative { sign_mask } else { 0 }; }
    let left_mantissa = if left_exp == 0 { left_fraction } else { (1 << fraction_bits) | left_fraction };
    let right_mantissa = if right_exp == 0 { right_fraction } else { (1 << fraction_bits) | right_fraction };
    // Each mantissa holds an implicit leading 1, so the product of two is scaled
    // by 2^(2*fraction_bits) relative to their integer values. Dividing by the
    // whole thing and not by a fixed shift is what keeps 1.0 * 1.0 at 1.0.
    let mut exponent = left_exp as i32 + right_exp as i32 - bias;
    let product = left_mantissa * right_mantissa;
    let mut mantissa = product >> fraction_bits;
    let remainder = product & fraction_mask;
    let half = 1u64 << (fraction_bits - 1);
    if remainder > half || (remainder == half && mantissa & 1 == 1) { mantissa += 1; }
    if mantissa >= (1 << (fraction_bits + 1)) { mantissa >>= 1; exponent += 1; }
    if exponent <= 0 { return if negative { sign_mask } else { 0 }; }
    if exponent >= max_exponent as i32 { return layout.overflow_word(negative); }
    (if negative { sign_mask } else { 0 }) | ((exponent as u64) << fraction_bits) | (mantissa & fraction_mask)
}

fn fp_div_bits(left: u64, right: u64, layout: Layout) -> u64 {
    let Layout { fraction_bits, bias, .. } = layout;
    let sign_mask = layout.sign_mask();
    let exponent_mask = layout.exponent_mask();
    let fraction_mask = layout.fraction_mask();
    let max_exponent = layout.max_exponent();
    let negative = layout.is_negative(left) != layout.is_negative(right);
    let left_exp = (left & exponent_mask) >> fraction_bits;
    let right_exp = (right & exponent_mask) >> fraction_bits;
    let left_fraction = left & fraction_mask;
    let right_fraction = right & fraction_mask;
    if layout.is_nan_word(left) || layout.is_nan_word(right) {
        return layout.nan_word();
    }
    if right_exp == 0 && right_fraction == 0 { return layout.overflow_word(negative); }
    if left_exp == 0 && left_fraction == 0 { return if negative { sign_mask } else { 0 }; }
    let left_mantissa = if left_exp == 0 { left_fraction } else { (1 << fraction_bits) | left_fraction };
    let right_mantissa = if right_exp == 0 { right_fraction } else { (1 << fraction_bits) | right_fraction };
    let mut exponent = left_exp as i32 - right_exp as i32 + bias;
    let mut quotient = (left_mantissa << (fraction_bits + 3)) / right_mantissa;
    while quotient >= (1 << (fraction_bits + 4)) { quotient >>= 1; exponent += 1; }
    while quotient < (1 << (fraction_bits + 3)) && exponent > 1 { quotient <<= 1; exponent -= 1; }
    if exponent <= 0 { return if negative { sign_mask } else { 0 }; }
    if exponent >= max_exponent as i32 { return layout.overflow_word(negative); }
    (if negative { sign_mask } else { 0 }) | ((exponent as u64) << fraction_bits) | ((quotient >> 3) & fraction_mask)
}

macro_rules! impl_float_bits {
    ($type:ty, $storage:ty, $byte_size:expr, $dtype:expr,
     $zero:expr, $one:expr, $half:expr, $two:expr,
     $neg_inf:expr, $pos_inf:expr, $epsilon:expr, $dtype_tag:expr,
     $exponent_bits:expr, $fraction_bits:expr, $bias:expr) => {
        // Comparing through the bit pattern would order -0.0 above +0.0 and
        // rank every negative below every positive, so the sign is split off
        // and the magnitudes compared on their own. Zeros collapse to one
        // value, and NaN is unordered — the reason `is_nan` exists next door.
        impl PartialEq for $type {
            fn eq(&self, other: &Self) -> bool {
                self.partial_cmp(other) == Some(core::cmp::Ordering::Equal)
            }
        }
        impl PartialOrd for $type {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                use core::cmp::Ordering;
                let layout = <$type as Float>::layout();
                let magnitude_mask = layout.magnitude_mask();
                // Everything below reads the packed form, where the format's
                // fields start at bit zero. TF32 is the reason: its 19 live bits
                // are the top of a 32-bit word, and packing is what makes one
                // set of masks right for it and for the others alike.
                let left = layout.pack(self.to_bits() as u64);
                let right = layout.pack(other.to_bits() as u64);
                if <Self as Float>::is_nan(*self) || <Self as Float>::is_nan(*other) {
                    return None;
                }
                let left_magnitude = left & magnitude_mask;
                let right_magnitude = right & magnitude_mask;
                // The two zeros differ by one bit and compare equal as values,
                // so the sign split below must not separate them.
                if left_magnitude == 0 && right_magnitude == 0 {
                    return Some(Ordering::Equal);
                }
                let left_negative = layout.is_negative(left);
                let right_negative = layout.is_negative(right);
                match (left_negative, right_negative) {
                    (false, true) => Some(Ordering::Greater),
                    (true, false) => Some(Ordering::Less),
                    (false, false) => left_magnitude.partial_cmp(&right_magnitude),
                    (true, true) => right_magnitude.partial_cmp(&left_magnitude),
                }
            }
        }
        impl core::ops::Add for $type {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                let layout = <$type as Float>::layout();
                Self::from_bits(layout.unpack(fp_add_bits(layout.pack(self.to_bits() as u64), layout.pack(rhs.to_bits() as u64), layout)) as _)
            }
        }
        impl core::ops::Sub for $type {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                let layout = <$type as Float>::layout();
                let flipped = flip_sign(layout.pack(rhs.to_bits() as u64), layout.sign_mask());
                Self::from_bits(layout.unpack(fp_add_bits(layout.pack(self.to_bits() as u64), flipped, layout)) as _)
            }
        }
        impl core::ops::Mul for $type {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                let layout = <$type as Float>::layout();
                Self::from_bits(layout.unpack(fp_mul_bits(layout.pack(self.to_bits() as u64), layout.pack(rhs.to_bits() as u64), layout)) as _)
            }
        }
        impl core::ops::Div for $type {
            type Output = Self;
            fn div(self, rhs: Self) -> Self {
                let layout = <$type as Float>::layout();
                Self::from_bits(layout.unpack(fp_div_bits(layout.pack(self.to_bits() as u64), layout.pack(rhs.to_bits() as u64), layout)) as _)
            }
        }
        impl core::ops::Neg for $type {
            type Output = Self;
            fn neg(self) -> Self {
                // Only the sign moves, and the sign is the one field that does
                // not move when a word is packed, so this needs no packing at
                // all: flip it where it actually sits.
                let layout = <$type as Float>::layout();
                Self::from_bits(((self.to_bits() as u64) ^ layout.storage_sign_mask()) as _)
            }
        }
        impl core::ops::AddAssign for $type { fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; } }
        impl core::ops::SubAssign for $type { fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; } }
        impl core::ops::MulAssign for $type { fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; } }
        impl core::ops::DivAssign for $type { fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; } }

        /// Build a value with the exact integer `n`, from this format's own
        /// exponent and fraction fields. No other format is involved: `n` is
        /// split into a biased exponent and a fraction right here, so the
        /// result is a native `$type` and never a narrowed neighbour.
        impl $type {
        const fn exact(n: u32) -> $type {
            if n == 0 { return <$type>::from_bits($zero); }
            let mut leading = 0u32;
            while (n >> (leading + 1)) != 0 { leading += 1; }
            let fraction_bits = $fraction_bits as u32;
            // `leading` counts bits below the implicit one, so the value is
            // 1.f * 2^leading and the fraction is n shifted to sit under the
            // implicit bit.
            let shift = fraction_bits.saturating_sub(leading);
            let fraction = (n << shift) & ((1u32 << fraction_bits) - 1);
            let biased = leading as i32 + $bias as i32;
            if biased >= (1 << $exponent_bits) - 1 {
                return <$type>::from_bits($pos_inf);
            }
            let layout = $dtype.layout();
            <$type>::from_bits(layout.unpack(((biased as u32) << fraction_bits | fraction) as u64) as _)
        }
        }

        impl Float for $type {
            const ZERO: Self = Self::from_bits($zero);
            const ONE: Self = Self::from_bits($one);
            const HALF: Self = Self::from_bits($half);
            const TWO: Self = Self::from_bits($two);
            const NEG_INF: Self = Self::from_bits($neg_inf);
            const POS_INF: Self = Self::from_bits($pos_inf);
            const EPSILON: Self = Self::from_bits($epsilon);

            // The sign is the one field that does not move when a word is
            // packed, so clearing it works on the stored word directly.
            fn abs(self) -> Self {
                let layout = <$type as Float>::layout();
                Self::from_bits(((self.to_bits() as u64) & !layout.storage_sign_mask()) as _)
            }
            // A saturated exponent means infinity and NaN only where the format
            // has an infinity. `E4M3` spends that field on real numbers, so there
            // the single top pattern is NaN and everything below it is finite.
            fn is_finite(self) -> bool {
                let layout = <$type as Float>::layout();
                layout.is_finite_word(layout.pack(self.to_bits() as u64))
            }
            fn is_nan(self) -> bool {
                let layout = <$type as Float>::layout();
                layout.is_nan_word(layout.pack(self.to_bits() as u64))
            }
            fn max(self, other: Self) -> Self {
                let layout = <$type as Float>::layout();
                let left = layout.pack(self.to_bits() as u64);
                let right = layout.pack(other.to_bits() as u64);
                let self_abs = left & layout.magnitude_mask();
                let other_abs = right & layout.magnitude_mask();
                if layout.is_nan_word(left) {
                    return other;
                }
                if layout.is_nan_word(right) {
                    return self;
                }
                if layout.is_negative(left) {
                    if !layout.is_negative(right) { other } else if self_abs < other_abs { other } else { self }
                } else if layout.is_negative(right) {
                    self
                } else if self_abs > other_abs { self } else { other }
            }
            fn min(self, other: Self) -> Self {
                let layout = <$type as Float>::layout();
                let left = layout.pack(self.to_bits() as u64);
                let right = layout.pack(other.to_bits() as u64);
                let self_abs = left & layout.magnitude_mask();
                let other_abs = right & layout.magnitude_mask();
                if layout.is_nan_word(left) {
                    return other;
                }
                if layout.is_nan_word(right) {
                    return self;
                }
                if layout.is_negative(left) {
                    if !layout.is_negative(right) { self } else if self_abs > other_abs { self } else { other }
                } else if layout.is_negative(right) {
                    other
                } else if self_abs < other_abs { self } else { other }
            }
            fn sqrt(self) -> Self {
                if self <= Self::ZERO { return Self::ZERO; }
                let mut estimate = if self > Self::ONE { self } else { Self::ONE };
                for _ in 0..8 { estimate = (estimate + self / estimate) / Self::TWO; }
                estimate
            }
            fn exp(self) -> Self {
                let mut term = Self::ONE;
                let mut sum = Self::ONE;
                for index in 1..16 {
                    term = term * self / Self::exact(index as u32);
                    sum += term;
                }
                sum
            }
            fn ln(self) -> Self {
                if self <= Self::ZERO { return Self::NEG_INF; }
                let y = (self - Self::ONE) / (self + Self::ONE);
                let y2 = y * y;
                let mut power = y;
                let mut sum = y;
                for index in 1..8 {
                    power *= y2;
                    sum += power / Self::exact((2 * index + 1) as u32);
                }
                Self::TWO * sum
            }
            fn sin(self) -> Self {
                let x2 = self * self;
                let mut term = self;
                let mut sum = self;
                for index in 1..6 {
                    let denominator = Self::exact(((2 * index) * (2 * index + 1)) as u32);
                    term = -term * x2 / denominator;
                    sum += term;
                }
                sum
            }
            fn cos(self) -> Self {
                let x2 = self * self;
                let mut term = Self::ONE;
                let mut sum = Self::ONE;
                for index in 1..6 {
                    let denominator = Self::exact(((2 * index - 1) * (2 * index)) as u32);
                    term = -term * x2 / denominator;
                    sum += term;
                }
                sum
            }
            fn tanh(self) -> Self {
                // `exp` of a negative argument is badly conditioned (the series
                // alternates), so the exponential is always fed a non-negative
                // value and the oddness of tanh restores the sign afterwards.
                let e = (Self::TWO * self.abs()).exp();
                let t = if e.is_finite() {
                    (e - Self::ONE) / (e + Self::ONE)
                } else {
                    Self::ONE
                };
                if self >= Self::ZERO { t } else { -t }
            }
            fn relu(self) -> Self {
                self.max(Self::ZERO)
            }
            fn sigmoid(self) -> Self {
                // Same conditioning trick as `tanh`: e^{|x|} is computed and
                // placed on the side the identity calls for, so neither branch
                // ever evaluates the series at a negative argument.
                let e = self.abs().exp();
                if !e.is_finite() {
                    return if self >= Self::ZERO { Self::ONE } else { Self::ZERO };
                }
                if self >= Self::ZERO {
                    e / (e + Self::ONE)
                } else {
                    Self::ONE / (e + Self::ONE)
                }
            }
            fn powf(self, other: Self) -> Self { (other * self.ln()).exp() }
            // Round to nearest, ties away from zero — the rule `roundf` follows
            // for f32, so a whole number here means the same as a whole number
            // there. Truncating the fraction instead would make 1.5 come back as
            // 1.0, which is a different function.
            fn round(self) -> Self {
                let layout = <$type as Float>::layout();
                let fraction_bits = layout.fraction_bits;
                let bits = layout.pack(self.to_bits() as u64);
                let exponent = ((bits >> fraction_bits) & layout.max_exponent()) as i32;
                if exponent == 0 || (layout.has_infinity && exponent == layout.max_exponent() as i32) {
                    return self;
                }
                // The value is 1.f * 2^unbiased, so the fraction starts
                // `shift` bits below the point and that many have to go.
                let shift = fraction_bits as i32 + layout.bias - exponent;
                if shift <= 0 { return self; }
                let half = 1u64 << (shift - 1);
                let dropped = bits & ((1u64 << shift) - 1);
                let mut result = bits & !((1u64 << shift) - 1);
                // Ties round away from zero, matching `roundf`/`roundd`: the
                // magnitude bits are incremented, and the sign is a separate
                // bit that is untouched, so a step up always grows |value|.
                if dropped >= half {
                    result += 1u64 << shift;
                }
                // The carry can reach the exponent field and saturate it. Where
                // the format has an infinity that is infinity; where it does not
                // the largest finite value is the same place to stop.
                if layout.has_infinity && (result & layout.exponent_mask()) == layout.exponent_mask() {
                    return Self::from_bits(layout.unpack((bits & layout.sign_mask()) | layout.exponent_mask()) as _);
                }
                if !layout.has_infinity && (result & layout.exponent_mask()) > layout.max_finite {
                    return Self::from_bits(layout.unpack((bits & layout.sign_mask()) | layout.max_finite) as _);
                }
                Self::from_bits(layout.unpack(result) as _)
            }
            fn recip(self) -> Self { Self::ONE / self }
            const BYTE_SIZE: usize = $byte_size;
            const DTYPE_TAG: u8 = $dtype_tag;
            fn layout() -> Layout { $dtype.layout() }
            fn dtype() -> DType { $dtype }
            fn to_f64_word(self) -> u64 { <$type as Float>::layout().pack(self.to_bits() as u64) }
            fn from_f64_word(word: u64) -> Self { Self::from_bits(<$type as Float>::layout().unpack(word) as _) }
            fn write_le_bytes(self, out: &mut [u8]) {
                out[..$byte_size].copy_from_slice(&self.to_bits().to_le_bytes());
            }
            fn from_le_bytes(bytes: &[u8]) -> Self {
                let mut word: $storage = 0;
                for index in 0..$byte_size {
                    word |= (bytes[index] as $storage) << (8 * index);
                }
                Self::from_bits(word)
            }
        }
    };
}

impl_float_bits!(F16, u16, 2, DType::F16,
    0x0000, 0x3c00, 0x3800, 0x4000, 0xfc00, 0x7c00, 0x1400, 2, 5, 10, 15);
impl_float_bits!(BF16, u16, 2, DType::BF16,
    0x0000, 0x3f80, 0x3f00, 0x4000, 0xff80, 0x7f80, 0x3c00, 3, 8, 7, 127);
// TF32 reads 32 bits but only 19 of them carry value: the same 8 exponent bits as
// F32, with the fraction cut to 10. Those 19 bits are the top of the word, so the
// constants below are F32's with the 13 dropped bits already zero — which is
// exactly how a checkpoint stores it, readable as F32 without moving anything.
// The epsilon moves with them: one step at 1.0 is 2^-10 here, not 2^-23.
impl_float_bits!(TF32, u32, 4, DType::TF32,
    0x0000_0000, 0x3f80_0000, 0x3f00_0000, 0x4000_0000,
    0xff80_0000, 0x7f80_0000, 0x3f80_2000, 4, 8, 10, 127);
// E4M3 has no infinity: the exponent field it would have spent on one holds real
// numbers, so 0x7E is a finite 448 and only 0x7F is NaN. That is why its layout
// says `has_infinity: false` — an overflow saturates to the largest finite value.
impl_float_bits!(E4M3, u8, 1, DType::E4M3,
    0x00, 0x38, 0x30, 0x40, 0xfe, 0x7e, 0x20, 5, 4, 3, 7);
// E5M2 is the other half of the FP8 pair: 5 exponent bits buy the range, 2
// fraction bits are all the precision, and it keeps a real infinity for the
// activations that overflow while the weights stay E4M3.
impl_float_bits!(E5M2, u8, 1, DType::E5M2,
    0x00, 0x3c, 0x38, 0x40, 0xfc, 0x7c, 0x34, 6, 5, 2, 15);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathError {
	Carve(CarveError),
	NonFinite,
}

impl From<CarveError> for MathError {
	fn from(error: CarveError) -> Self {
		MathError::Carve(error)
	}
}

pub(crate) fn carve_buffer<'a, T: Float>(
	carve: Carve<'a>,
) -> Result<Buf<'a, T>, MathError> {
	Buf::from_carve(carve).map_err(MathError::from)
}

pub(crate) fn carve_softmax_rows<T: Float>(
	buffer: &mut Buf<'_, T>,
	rows: usize,
) -> Result<usize, MathError> {
	let count = T::carve_count(buffer)?;

	if count == 0 {
		return Ok(0);
	}

	if rows == 0 || count % rows != 0 {
		return Err(CarveError::SizeMismatch.into());
	}

	let width = count / rows;

	for row in 0..rows {
		let start = row * width;
		let end = start + width;

		let mut peak = T::NEG_INF;

		for index in start..end {
			let value = T::carve_read(buffer, index)?;

			if value > peak {
				peak = value;
			}
		}

		let mut total = T::ZERO;

		for index in start..end {
			let value = T::carve_read(buffer, index)?;
			let exponent = (value - peak).exp();

			if !exponent.is_finite() {
				return Err(MathError::NonFinite);
			}

			total += exponent;
		}

		if !total.is_finite() || total <= T::ZERO {
			return Err(MathError::NonFinite);
		}

		let inverse = T::ONE / total;

		for index in start..end {
			let value = T::carve_read(buffer, index)?;
			let normalized = (value - peak).exp() * inverse;

			if !normalized.is_finite() {
				return Err(MathError::NonFinite);
			}

			T::carve_write(buffer, index, normalized)?;
		}
	}

	Ok(count)
}
