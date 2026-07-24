use super::exp_log::{expd, expf, lnd, lnf};
use super::power::{powd, powf};
use super::roots::{sqrtd, sqrtf};
use super::rounding::{roundd, roundf};
use super::trig::{cosd, cosf, sind, sinf, tanhd, tanhf};

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

    fn from_usize(n: usize) -> Self;
    fn from_i32(n: i32) -> Self;
    fn from_f32(n: f32) -> Self;
    fn from_f64(n: f64) -> Self;
    fn to_f64(self) -> f64;

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
    fn powf(self, other: Self) -> Self;
    fn round(self) -> Self;
    fn recip(self) -> Self;

    const BYTE_SIZE: usize;
    const DTYPE_TAG: u8;
    fn write_le_bytes(self, out: &mut [u8]);
    fn from_le_bytes(bytes: &[u8]) -> Self;
}

impl Float for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const NEG_INF: Self = f32::NEG_INFINITY;
    const POS_INF: Self = f32::INFINITY;
    const EPSILON: Self = f32::EPSILON;

    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
    #[inline]
    fn from_i32(n: i32) -> Self {
        n as f32
    }
    #[inline]
    fn from_f32(n: f32) -> Self {
        n
    }
    #[inline]
    fn from_f64(n: f64) -> Self {
        n as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn abs(self) -> Self {
        if self < 0.0 {
            -self
        } else {
            self
        }
    }
    #[inline]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
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
        sqrtf(self)
    }
    #[inline]
    fn exp(self) -> Self {
        expf(self)
    }
    #[inline]
    fn ln(self) -> Self {
        lnf(self)
    }
    #[inline]
    fn sin(self) -> Self {
        sinf(self)
    }
    #[inline]
    fn cos(self) -> Self {
        cosf(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        tanhf(self)
    }
    #[inline]
    fn powf(self, other: Self) -> Self {
        powf(self, other)
    }
    #[inline]
    fn round(self) -> Self {
        roundf(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    const BYTE_SIZE: usize = 4;
    const DTYPE_TAG: u8 = 0;
    #[inline]
    fn write_le_bytes(self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl Float for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;
    const TWO: Self = 2.0;
    const NEG_INF: Self = f64::NEG_INFINITY;
    const POS_INF: Self = f64::INFINITY;
    const EPSILON: Self = f64::EPSILON;

    #[inline]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
    #[inline]
    fn from_i32(n: i32) -> Self {
        n as f64
    }
    #[inline]
    fn from_f32(n: f32) -> Self {
        n as f64
    }
    #[inline]
    fn from_f64(n: f64) -> Self {
        n
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn abs(self) -> Self {
        if self < 0.0 {
            -self
        } else {
            self
        }
    }
    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
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
        sqrtd(self)
    }
    #[inline]
    fn exp(self) -> Self {
        expd(self)
    }
    #[inline]
    fn ln(self) -> Self {
        lnd(self)
    }
    #[inline]
    fn sin(self) -> Self {
        sind(self)
    }
    #[inline]
    fn cos(self) -> Self {
        cosd(self)
    }
    #[inline]
    fn tanh(self) -> Self {
        tanhd(self)
    }
    #[inline]
    fn powf(self, other: Self) -> Self {
        powd(self, other)
    }
    #[inline]
    fn round(self) -> Self {
        roundd(self)
    }
    #[inline]
    fn recip(self) -> Self {
        1.0 / self
    }

    const BYTE_SIZE: usize = 8;
    const DTYPE_TAG: u8 = 1;
    #[inline]
    fn write_le_bytes(self, out: &mut [u8]) {
        out[..8].copy_from_slice(&self.to_le_bytes());
    }
    #[inline]
    fn from_le_bytes(bytes: &[u8]) -> Self {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}
