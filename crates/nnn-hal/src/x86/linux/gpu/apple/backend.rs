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
    VendorBackendConfig {
        workgroup_size: 32,
        compute_queues: 3,
        render_threads: 16,
        double_buffered: true,
        frame_budget_us: crate::arch::detected_frame_budget_us(),
        low_power: false,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    requested.clamp(1, 32)
}
