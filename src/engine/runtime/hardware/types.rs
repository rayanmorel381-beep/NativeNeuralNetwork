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

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainingRuntimePlan {
    pub batch_size: usize,
    pub max_workers: usize,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct BackendCapabilities {
    pub opcode_mask: u64,
    pub sustained_flops_per_second: f64,
    pub total_memory_bytes: usize,
    pub available_memory_bytes: usize,
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

pub struct SwapBatch {
    ptr: *mut u8,
    bytes: usize,
    input_size: usize,
    output_size: usize,
    count: usize,
    max_count: usize,
}

unsafe impl Send for SwapBatch {}
unsafe impl Sync for SwapBatch {}

impl SwapBatch {
    pub fn new(input_size: usize, output_size: usize, max_count: usize) -> Option<Self> {
        let sample_bytes = (input_size.saturating_add(output_size)).saturating_mul(4);
        let bytes = sample_bytes.saturating_mul(max_count);
        if bytes == 0 {
            return None;
        }
        let ptr = crate::engine::runtime::hardware::mmap_shared_anon(bytes);
        if ptr.is_null() {
            return None;
        }
        Some(Self { ptr, bytes, input_size, output_size, count: 0, max_count })
    }

    pub fn input_size(&self) -> usize { self.input_size }
    pub fn output_size(&self) -> usize { self.output_size }
    pub fn sample_bytes(&self) -> usize { (self.input_size + self.output_size) * 4 }
    pub fn max_count(&self) -> usize { self.max_count }
    pub fn set_count(&mut self, count: usize) { self.count = count.min(self.max_count); }
    pub fn ptr(&mut self) -> *mut u8 { self.ptr }

    pub fn get_sample(&self, i: usize) -> (&[f32], &[f32]) {
        let sample_bytes = self.sample_bytes();
        let offset = i * sample_bytes;
        unsafe {
            let inp = core::slice::from_raw_parts(self.ptr.add(offset) as *const f32, self.input_size);
            let tgt = core::slice::from_raw_parts(self.ptr.add(offset + self.input_size * 4) as *const f32, self.output_size);
            (inp, tgt)
        }
    }

    pub fn release(&mut self) {
        if !self.ptr.is_null() {
            crate::engine::runtime::hardware::munmap(self.ptr, self.bytes);
            self.ptr = core::ptr::null_mut();
        }
    }
}
