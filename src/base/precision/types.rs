#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Precision {
    F64 = 0,
    F32 = 1,
    F16 = 2,
    BF16 = 3,
}
