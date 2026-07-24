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
    let total = crate::engine::runtime::hardware::arm::detected_parallelism();
    VendorBackendConfig {
        workgroup_size: 128,
        compute_queues: total.min(8),
        render_threads: total,
        double_buffered: true,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: false,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let max_workers = crate::engine::runtime::hardware::arm::detected_parallelism();
    requested.max(1).min(max_workers)
}
