#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tick(pub u64);

impl Tick {
    pub(crate) fn advance(self, delay: u32) -> Tick {
        Tick(self.0.wrapping_add(delay as u64))
    }
}
