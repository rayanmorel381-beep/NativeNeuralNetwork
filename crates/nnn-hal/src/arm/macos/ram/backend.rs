#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) page_size: usize,
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: Option<u64>,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let page = super::super::host_page_size();
    let total_bytes = super::super::sysctl_u64(b"hw.memsize\0").unwrap_or(0);
    let available_bytes = super::super::sysctl_u64(b"hw.usermem\0");
    let total_gb = total_bytes / (1024 * 1024 * 1024);
    VendorBackendConfig {
        page_size: page,
        total_bytes,
        available_bytes,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: total_gb <= 8,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    requested.max(1)
}
