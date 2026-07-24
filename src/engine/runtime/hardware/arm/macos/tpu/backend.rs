#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) workgroup_size: usize,
    pub(crate) compute_queues: usize,
    pub(crate) render_threads: usize,
    pub(crate) double_buffered: bool,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let p_cores = super::super::p_core_count();
    let e_cores = super::super::e_core_count();
    let total = (p_cores + e_cores).max(1);
    let mem_bytes = super::super::unified_memory_bytes();
    let mem_gb = mem_bytes / (1024 * 1024 * 1024);
    let compute_queues = if p_cores >= 8 { 4 } else { 2 };
    VendorBackendConfig {
        workgroup_size: 128,
        compute_queues: compute_queues.min(total),
        render_threads: total,
        double_buffered: true,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: mem_gb <= 8,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let p_cores = super::super::p_core_count().max(1);
    requested.max(1).min(p_cores)
}
