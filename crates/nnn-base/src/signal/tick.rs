/// A logical time step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tick(pub u64);

impl Tick {
    /// This tick shifted by `delay` steps, saturating on overflow.
    pub fn advance(self, delay: u32) -> Tick {
        Tick(self.0.saturating_add(delay as u64))
    }
}
