#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RuntimeProfile {
    pub batch_size: usize,
    pub sequence_len: usize,
    pub bytes_per_param: usize,
    pub bytes_per_activation: usize,
    pub bytes_per_kv: usize,
}

impl RuntimeProfile {
    pub const fn fp32(batch_size: usize, sequence_len: usize) -> Self {
        Self { batch_size, sequence_len, bytes_per_param: 4, bytes_per_activation: 4, bytes_per_kv: 4 }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HardwareProfile {
    pub cores: usize,
    pub ram_total: usize,
    pub ram_available: usize,
    pub gpu: bool,
    pub tpu: bool,
    pub lpu: bool,
    pub cpu_avg_mhz: u32,
    pub cpu_max_mhz: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HardwareComponentSnapshot {
    pub present: u8,
    pub reserved0: u8,
    pub reserved1: u8,
    pub reserved2: u8,
    pub device_count: u32,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub sustained_flops_per_second: u64,
    pub kernel_spinlock_flags: u64,
    pub clock_hz: u64,
    pub cycle_counter_hz: u64,
    pub nominal_cycles_per_kernel: u64,
    pub observed_cycles_per_kernel: u64,
    pub timing_flags: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HardwareDetectionSnapshot {
    pub cpu_logical_cores: u32,
    pub os_flags: u32,
    pub hardware_flags: u32,
    pub system_total_memory_bytes: u64,
    pub system_available_memory_bytes: u64,
    pub cpu_kernel_spinlock_flags: u64,
    pub cpu_clock_hz: u64,
    pub cpu_cycle_counter_hz: u64,
    pub cpu_nominal_cycles_per_kernel: u64,
    pub cpu_observed_cycles_per_kernel: u64,
    pub cpu_timing_flags: u64,
    pub gpu: HardwareComponentSnapshot,
    pub tpu: HardwareComponentSnapshot,
    pub lpu: HardwareComponentSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendContractCapabilities {
    pub magic: u32,
    pub abi_version: u16,
    pub reserved: u16,
    pub opcode_mask: u32,
    pub feature_flags: u32,
    pub max_batch: u32,
    pub max_elements: u32,
    pub preferred_alignment: u32,
    pub sustained_flops_per_second: u64,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamAbstraction {
    pub total_bytes: u128,
    pub available_bytes: u128,
    pub used_bytes: u128,
    pub total_mib: u128,
    pub available_mib: u128,
    pub used_mib: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct BackendCapabilities {
    pub opcode_mask: u64,
    pub sustained_flops_per_second: f64,
    pub total_memory_bytes: usize,
    pub available_memory_bytes: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainingRuntimePlan {
    pub batch_size: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ResourceSnapshot {
    pub cpu_usage: f32,
    pub ram_total: usize,
    pub ram_available: usize,
    pub gpu_dispatches: usize,
    pub tpu_dispatches: usize,
    pub lpu_dispatches: usize,
}

impl ResourceSnapshot {
    pub fn cpu_load(&self) -> f32 {
        self.cpu_usage
    }

    pub fn memory_budget(&self) -> usize {
        self.ram_available.max(self.ram_total).max(1)
    }

    pub fn accelerator_dispatches(&self) -> usize {
        self.gpu_dispatches + self.tpu_dispatches + self.lpu_dispatches
    }
}

impl BackendCapabilities {
    pub fn opcode_bits(&self) -> u64 {
        self.opcode_mask
    }

    pub fn flops(&self) -> f64 {
        self.sustained_flops_per_second
    }

    pub fn memory_budget_bytes(&self) -> usize {
        self.available_memory_bytes.max(self.total_memory_bytes)
    }
}

