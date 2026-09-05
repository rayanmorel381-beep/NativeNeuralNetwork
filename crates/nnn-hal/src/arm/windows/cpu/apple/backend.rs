#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) worker_hint: usize,
    pub(crate) render_workers: usize,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let info = super::detect_arm();
    let total = info.as_ref().map_or_else(
        crate::engine::runtime::hardware::arm::detected_parallelism,
        |i| i.core_count as usize,
    );
    let low_power = info.as_ref().is_some_and(|i| {
        i.brand_contains_ignore_case(b"lite") || i.brand_contains_ignore_case(b"nano") || i.core_count < 4
    });
    let p_cores = (total / 2).max(1);
    let render_workers = p_cores.saturating_sub(1).max(1);
    VendorBackendConfig {
        worker_hint: p_cores,
        render_workers,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let total = super::detect_arm()
        .as_ref()
        .map_or_else(
            crate::engine::runtime::hardware::arm::detected_parallelism,
            |i| i.core_count as usize,
        );
    let p_cores = (total / 2).max(1);
    requested.max(1).min(p_cores)
}
