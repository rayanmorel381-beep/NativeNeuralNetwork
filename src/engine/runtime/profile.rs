#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeProfile {
    pub batch_size: usize,
    pub sequence_len: usize,
    pub bytes_per_param: usize,
    pub bytes_per_activation: usize,
    pub bytes_per_kv: usize,
}

impl RuntimeProfile {
    pub const fn fp32(batch_size: usize, sequence_len: usize) -> Self {
        Self {
            batch_size,
            sequence_len,
            bytes_per_param: 4,
            bytes_per_activation: 4,
            bytes_per_kv: 4,
        }
    }

    pub const fn fp16(batch_size: usize, sequence_len: usize) -> Self {
        Self {
            batch_size,
            sequence_len,
            bytes_per_param: 2,
            bytes_per_activation: 2,
            bytes_per_kv: 2,
        }
    }

    pub fn validate(&self) -> bool {
        self.batch_size > 0
            && self.sequence_len > 0
            && self.bytes_per_param > 0
            && self.bytes_per_activation > 0
            && self.bytes_per_kv > 0
    }

    pub fn token_count(&self) -> Option<usize> {
        self.batch_size.checked_mul(self.sequence_len)
    }
}
