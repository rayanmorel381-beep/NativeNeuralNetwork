use core::ops::Range;
use core::str;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bytes<'a> {
    data: &'a [u8],
}

impl<'a> Bytes<'a> {
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    pub const fn len(&self) -> usize {
        self.data.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub const fn as_slice(&self) -> &'a [u8] {
        self.data
    }

    pub fn slice(&self, start: usize, end: usize) -> Option<&'a [u8]> {
        self.data.get(start..end)
    }

    pub fn u8_at(&self, offset: usize) -> Option<u8> {
        self.data.get(offset).copied()
    }

    pub fn i8_at(&self, offset: usize) -> Option<i8> {
        self.u8_at(offset).map(|value| value as i8)
    }

    pub fn u16_at(&self, offset: usize) -> Option<u16> {
        Some(u16::from_le_bytes(self.array(offset)?))
    }

    pub fn i16_at(&self, offset: usize) -> Option<i16> {
        Some(i16::from_le_bytes(self.array(offset)?))
    }

    pub fn u32_at(&self, offset: usize) -> Option<u32> {
        Some(u32::from_le_bytes(self.array(offset)?))
    }

    pub fn i32_at(&self, offset: usize) -> Option<i32> {
        Some(i32::from_le_bytes(self.array(offset)?))
    }

    pub fn u64_at(&self, offset: usize) -> Option<u64> {
        Some(u64::from_le_bytes(self.array(offset)?))
    }

    pub fn i64_at(&self, offset: usize) -> Option<i64> {
        Some(i64::from_le_bytes(self.array(offset)?))
    }

    pub fn f32_at(&self, offset: usize) -> Option<f32> {
        Some(f32::from_bits(self.u32_at(offset)?))
    }

    pub fn f64_at(&self, offset: usize) -> Option<f64> {
        Some(f64::from_bits(self.u64_at(offset)?))
    }

    pub fn u16_be_at(&self, offset: usize) -> Option<u16> {
        Some(u16::from_be_bytes(self.array(offset)?))
    }

    pub fn i16_be_at(&self, offset: usize) -> Option<i16> {
        Some(i16::from_be_bytes(self.array(offset)?))
    }

    pub fn u32_be_at(&self, offset: usize) -> Option<u32> {
        Some(u32::from_be_bytes(self.array(offset)?))
    }

    pub fn i32_be_at(&self, offset: usize) -> Option<i32> {
        Some(i32::from_be_bytes(self.array(offset)?))
    }

    pub fn u64_be_at(&self, offset: usize) -> Option<u64> {
        Some(u64::from_be_bytes(self.array(offset)?))
    }

    pub fn i64_be_at(&self, offset: usize) -> Option<i64> {
        Some(i64::from_be_bytes(self.array(offset)?))
    }

    pub fn f32_be_at(&self, offset: usize) -> Option<f32> {
        Some(f32::from_bits(self.u32_be_at(offset)?))
    }

    pub fn f64_be_at(&self, offset: usize) -> Option<f64> {
        Some(f64::from_bits(self.u64_be_at(offset)?))
    }

    pub fn str_at(&self, start: usize, len: usize) -> Option<&'a str> {
        str::from_utf8(self.slice(start, start.checked_add(len)?)?).ok()
    }

    fn array<const N: usize>(&self, offset: usize) -> Option<[u8; N]> {
        let end = offset.checked_add(N)?;
        self.data.get(offset..end)?.try_into().ok()
    }
}

#[derive(Debug)]
pub struct BytesMut<'a> {
    data: &'a mut [u8],
}

impl<'a> BytesMut<'a> {
    pub fn new(data: &'a mut [u8]) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn as_slice(&self) -> &[u8] {
        self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.data
    }

    pub fn as_bytes(&self) -> Bytes<'_> {
        Bytes::new(self.data)
    }

    pub fn slice(&self, start: usize, end: usize) -> Option<&[u8]> {
        self.data.get(start..end)
    }

    pub fn slice_mut(&mut self, start: usize, end: usize) -> Option<&mut [u8]> {
        self.data.get_mut(start..end)
    }

    pub fn fill(&mut self, value: u8) {
        self.data.fill(value);
    }

    pub fn copy_from_slice(&mut self, offset: usize, src: &[u8]) -> bool {
        let Some(end) = offset.checked_add(src.len()) else {
            return false;
        };
        match self.data.get_mut(offset..end) {
            Some(dst) => {
                dst.copy_from_slice(src);
                true
            }
            None => false,
        }
    }

    pub fn copy_within(&mut self, src: Range<usize>, dest: usize) -> bool {
        if src.start > src.end {
            return false;
        }
        let Some(end) = dest.checked_add(src.end - src.start) else {
            return false;
        };
        if src.end > self.data.len() || end > self.data.len() {
            return false;
        }
        self.data.copy_within(src, dest);
        true
    }

    pub fn set_u8(&mut self, offset: usize, value: u8) -> bool {
        self.write(offset, [value])
    }

    pub fn set_i8(&mut self, offset: usize, value: i8) -> bool {
        self.write(offset, [value as u8])
    }

    pub fn set_u16_le(&mut self, offset: usize, value: u16) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_u16_be(&mut self, offset: usize, value: u16) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_i16_le(&mut self, offset: usize, value: i16) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_i16_be(&mut self, offset: usize, value: i16) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_u32_le(&mut self, offset: usize, value: u32) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_u32_be(&mut self, offset: usize, value: u32) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_i32_le(&mut self, offset: usize, value: i32) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_i32_be(&mut self, offset: usize, value: i32) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_u64_le(&mut self, offset: usize, value: u64) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_u64_be(&mut self, offset: usize, value: u64) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_i64_le(&mut self, offset: usize, value: i64) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_i64_be(&mut self, offset: usize, value: i64) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_f32_le(&mut self, offset: usize, value: f32) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_f32_be(&mut self, offset: usize, value: f32) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    pub fn set_f64_le(&mut self, offset: usize, value: f64) -> bool {
        self.write(offset, value.to_le_bytes())
    }

    pub fn set_f64_be(&mut self, offset: usize, value: f64) -> bool {
        self.write(offset, value.to_be_bytes())
    }

    fn write<const N: usize>(&mut self, offset: usize, bytes: [u8; N]) -> bool {
        let Some(end) = offset.checked_add(N) else {
            return false;
        };
        match self.data.get_mut(offset..end) {
            Some(dst) => {
                dst.copy_from_slice(&bytes);
                true
            }
            None => false,
        }
    }
}
