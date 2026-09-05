use super::{amd, apple, intel};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vendor {
    Amd,
    Intel,
    Apple,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RamConfig {
    pub(crate) vendor: Vendor,
    pub(crate) page_size: usize,
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: Option<u64>,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RamSchedule {
    pub(crate) chunks: usize,
    pub(crate) chunk_size: usize,
    pub(crate) frame_budget_us: u64,
}

fn detect_vendor() -> Vendor {
    if let Some(id) = crate::engine::runtime::hardware::x86::process_identifier() {
        if id.contains("apple") {
            return Vendor::Apple;
        }
        if id.contains("amd") {
            return Vendor::Amd;
        }
        if id.contains("intel") {
            return Vendor::Intel;
        }
    }
    Vendor::Apple
}

pub(crate) fn default_config() -> RamConfig {
    let vendor = detect_vendor();
    let (page_size, total_bytes, available_bytes, frame_budget_us, low_power) = match vendor {
        Vendor::Amd => {
            let c = amd::backend::default_backend_config();
            (c.page_size, c.total_bytes, c.available_bytes, c.frame_budget_us, c.low_power)
        }
        Vendor::Intel => {
            let c = intel::backend::default_backend_config();
            (c.page_size, c.total_bytes, c.available_bytes, c.frame_budget_us, c.low_power)
        }
        Vendor::Apple => {
            let c = apple::backend::default_backend_config();
            (c.page_size, c.total_bytes, c.available_bytes, c.frame_budget_us, c.low_power)
        }
    };
    RamConfig { vendor, page_size, total_bytes, available_bytes, frame_budget_us, low_power }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    match detect_vendor() {
        Vendor::Amd => amd::backend::clamp_workers(requested),
        Vendor::Intel => intel::backend::clamp_workers(requested),
        Vendor::Apple => apple::backend::clamp_workers(requested),
    }
}

pub(crate) fn build_schedule(work_items: usize) -> RamSchedule {
    let vendor = detect_vendor();
    let (chunks, chunk_size, frame_budget_us) = match vendor {
        Vendor::Amd => {
            let s = amd::scheduler::build_schedule(work_items);
            (s.chunks, s.chunk_size, s.frame_budget_us)
        }
        Vendor::Intel => {
            let s = intel::scheduler::build_schedule(work_items);
            (s.chunks, s.chunk_size, s.frame_budget_us)
        }
        Vendor::Apple => {
            let s = apple::scheduler::build_schedule(work_items);
            (s.chunks, s.chunk_size, s.frame_budget_us)
        }
    };
    RamSchedule { chunks, chunk_size, frame_budget_us }
}
