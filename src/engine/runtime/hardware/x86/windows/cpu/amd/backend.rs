#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) worker_hint: usize,
    pub(crate) render_workers: usize,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let total = crate::engine::runtime::hardware::x86::detected_parallelism();
    let render_workers = total.saturating_sub(1).max(1);
    VendorBackendConfig {
        worker_hint: total,
        render_workers,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: false,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let max_workers = crate::engine::runtime::hardware::x86::detected_parallelism();
    requested.max(1).min(max_workers)
}
