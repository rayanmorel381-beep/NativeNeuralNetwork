#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpCounter {
    pub matmul_ops: u64,
    pub attention_ops: u64,
    pub activation_ops: u64,
    pub normalization_ops: u64,
    pub memory_read_bytes: u64,
    pub memory_write_bytes: u64,
}

impl OpCounter {
    pub const fn new() -> Self {
        Self {
            matmul_ops: 0,
            attention_ops: 0,
            activation_ops: 0,
            normalization_ops: 0,
            memory_read_bytes: 0,
            memory_write_bytes: 0,
        }
    }

    pub fn add_matmul(&mut self, m: usize, n: usize, k: usize) {
        let ops = (m as u64)
            .saturating_mul(n as u64)
            .saturating_mul(k as u64)
            .saturating_mul(2);
        self.matmul_ops = self.matmul_ops.saturating_add(ops);
    }

    pub fn add_attention(&mut self, q: usize, k: usize, d: usize) {
        let ops = (q as u64)
            .saturating_mul(k as u64)
            .saturating_mul(d as u64)
            .saturating_mul(2);
        self.attention_ops = self.attention_ops.saturating_add(ops);
    }

    pub fn add_activation(&mut self, elems: usize) {
        self.activation_ops = self.activation_ops.saturating_add(elems as u64);
    }

    pub fn add_normalization(&mut self, elems: usize) {
        let ops = (elems as u64).saturating_mul(6);
        self.normalization_ops = self.normalization_ops.saturating_add(ops);
    }

    pub fn add_memory_read(&mut self, bytes: usize) {
        self.memory_read_bytes = self.memory_read_bytes.saturating_add(bytes as u64);
    }

    pub fn add_memory_write(&mut self, bytes: usize) {
        self.memory_write_bytes = self.memory_write_bytes.saturating_add(bytes as u64);
    }

    pub fn merge(&mut self, other: &OpCounter) {
        self.matmul_ops = self.matmul_ops.saturating_add(other.matmul_ops);
        self.attention_ops = self.attention_ops.saturating_add(other.attention_ops);
        self.activation_ops = self.activation_ops.saturating_add(other.activation_ops);
        self.normalization_ops = self
            .normalization_ops
            .saturating_add(other.normalization_ops);
        self.memory_read_bytes = self
            .memory_read_bytes
            .saturating_add(other.memory_read_bytes);
        self.memory_write_bytes = self
            .memory_write_bytes
            .saturating_add(other.memory_write_bytes);
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    pub fn total_ops(&self) -> u64 {
        self.matmul_ops
            .saturating_add(self.attention_ops)
            .saturating_add(self.activation_ops)
            .saturating_add(self.normalization_ops)
    }

    pub fn total_memory_bytes(&self) -> u64 {
        self.memory_read_bytes
            .saturating_add(self.memory_write_bytes)
    }
}

impl Default for OpCounter {
    fn default() -> Self {
        Self::new()
    }
}
