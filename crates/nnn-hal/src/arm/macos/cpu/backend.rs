#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct AppleSiliconCpuInfo {
    pub p_cores: u8,
    pub e_cores: u8,
    pub freq_p_max_hz: u64,
    pub freq_e_max_hz: u64,
    pub l2_cache_bytes: u64,
    pub l3_cache_bytes: u64,
    pub total_logical: u8,
}

pub(crate) fn detect() -> AppleSiliconCpuInfo {
    let p_cores = super::sysctl_u64(b"hw.perflevel0.physicalcpu\0")
        .filter(|&n| n > 0)
        .unwrap_or(1) as u8;
    let e_cores = super::sysctl_u64(b"hw.perflevel1.physicalcpu\0")
        .unwrap_or(0) as u8;
    let freq_p_max_hz = super::sysctl_u64(b"hw.perflevel0.cpufrequency_max\0")
        .or_else(|| super::sysctl_u64(b"hw.cpufrequency_max\0"))
        .unwrap_or(0);
    let freq_e_max_hz = super::sysctl_u64(b"hw.perflevel1.cpufrequency_max\0")
        .unwrap_or(0);
    let l2_cache_bytes = super::sysctl_u64(b"hw.l2cachesize\0").unwrap_or(0);
    let l3_cache_bytes = super::sysctl_u64(b"hw.l3cachesize\0").unwrap_or(0);
    let total_logical = super::sysctl_u64(b"hw.logicalcpu\0")
        .filter(|&n| n > 0)
        .unwrap_or(4) as u8;
    AppleSiliconCpuInfo {
        p_cores,
        e_cores,
        freq_p_max_hz,
        freq_e_max_hz,
        l2_cache_bytes,
        l3_cache_bytes,
        total_logical,
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) p_core_workers: usize,
    pub(crate) e_core_workers: usize,
    pub(crate) render_workers: usize,
    pub(crate) freq_p_max_hz: u64,
    pub(crate) freq_e_max_hz: u64,
    pub(crate) l2_cache_bytes: u64,
    pub(crate) l3_cache_bytes: u64,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let info = detect();
    let p = info.p_cores as usize;
    let e = info.e_cores as usize;
    let total = (p + e).max(info.total_logical as usize).max(1);
    VendorBackendConfig {
        p_core_workers: p,
        e_core_workers: e,
        render_workers: p.saturating_sub(1).max(1),
        freq_p_max_hz: info.freq_p_max_hz,
        freq_e_max_hz: info.freq_e_max_hz,
        l2_cache_bytes: info.l2_cache_bytes,
        l3_cache_bytes: info.l3_cache_bytes,
        frame_budget_us: crate::arch::detected_frame_budget_us(),
        low_power: total <= 6,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let info = detect();
    let max = (info.p_cores as usize).max(1);
    requested.max(1).min(max)
}
