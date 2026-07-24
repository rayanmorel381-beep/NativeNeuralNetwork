#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningMean {
    pub sum: f32,
    pub count: u64,
}

impl RunningMean {
    pub const fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }

    pub fn update(&mut self, value: f32) {
        self.sum += value;
        self.count = self.count.saturating_add(1);
    }

    pub fn merge(&mut self, other: RunningMean) {
        self.sum += other.sum;
        self.count = self.count.saturating_add(other.count);
    }

    pub fn value(&self) -> Option<f32> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f32)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningMeanF64 {
    pub sum: f64,
    pub count: u64,
}

impl RunningMeanF64 {
    pub const fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }

    pub fn update(&mut self, value: f64) {
        self.sum += value;
        self.count = self.count.saturating_add(1);
    }

    pub fn merge(&mut self, other: RunningMeanF64) {
        self.sum += other.sum;
        self.count = self.count.saturating_add(other.count);
    }

    pub fn value(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.sum / self.count as f64)
        }
    }
}

impl Default for RunningMeanF64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RunningMean {
    fn default() -> Self {
        Self::new()
    }
}
