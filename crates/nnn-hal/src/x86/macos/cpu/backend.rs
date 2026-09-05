unsafe extern "C" {
    fn sysctlbyname(
        name: *const u8,
        oldp: *mut u8,
        oldlenp: *mut usize,
        newp: *const u8,
        newlen: usize,
    ) -> i32;
}

fn sysctl_u64(name: &[u8]) -> Option<u64> {
    let mut out: u64 = 0;
    let mut len = core::mem::size_of::<u64>();
    let ret = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut out as *mut u64 as *mut u8,
            &mut len,
            core::ptr::null(),
            0,
        )
    };
    if ret == 0 { Some(out) } else { None }
}

pub(crate) struct AppleSiliconCpuInfo {
    pub p_cores: u8,
    pub e_cores: u8,
    pub freq_p_max_hz: u64,
    pub freq_e_max_hz: u64,
    pub l2_cache_bytes: u64,
}

pub(crate) fn detect() -> AppleSiliconCpuInfo {
    let p_cores = sysctl_u64(b"hw.perflevel0.physicalcpu\0").unwrap_or(4).max(1) as u8;
    let e_cores = sysctl_u64(b"hw.perflevel1.physicalcpu\0").unwrap_or(4) as u8;
    let freq_p_max_hz = sysctl_u64(b"hw.perflevel0.cpufrequency_max\0").unwrap_or(0);
    let freq_e_max_hz = sysctl_u64(b"hw.perflevel1.cpufrequency_max\0").unwrap_or(0);
    let l2_cache_bytes = sysctl_u64(b"hw.l2cachesize\0").unwrap_or(0);
    AppleSiliconCpuInfo { p_cores, e_cores, freq_p_max_hz, freq_e_max_hz, l2_cache_bytes }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) p_core_workers: usize,
    pub(crate) e_core_workers: usize,
    pub(crate) render_workers: usize,
    pub(crate) freq_p_max_hz: u64,
    pub(crate) freq_e_max_hz: u64,
    pub(crate) l2_cache_bytes: u64,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let info = detect();
    let p_cores = info.p_cores as usize;
    let e_cores = info.e_cores as usize;
    let total = p_cores + e_cores;
    let render_workers = p_cores.saturating_sub(1).max(1);
    VendorBackendConfig {
        p_core_workers: p_cores,
        e_core_workers: e_cores,
        render_workers,
        freq_p_max_hz: info.freq_p_max_hz,
        freq_e_max_hz: info.freq_e_max_hz,
        l2_cache_bytes: info.l2_cache_bytes,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: total <= 6,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let p_cores = sysctl_u64(b"hw.perflevel0.physicalcpu\0").unwrap_or(1).max(1) as usize;
    requested.max(1).min(p_cores)
}
