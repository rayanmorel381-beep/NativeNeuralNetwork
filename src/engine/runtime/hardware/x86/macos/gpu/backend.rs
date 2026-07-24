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

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) gpu_cores: usize,
    pub(crate) metal_queues: usize,
    pub(crate) tile_size: usize,
    pub(crate) simd_width: usize,
    pub(crate) tile_memory_bytes: usize,
    pub(crate) unified_memory_bytes: u64,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let gpu_cores = sysctl_u64(b"hw.gpu.core_count\0").unwrap_or(8) as usize;
    let unified_memory_bytes = sysctl_u64(b"hw.memsize\0").unwrap_or(0);
    let unified_mb = unified_memory_bytes / (1024 * 1024);
    let metal_queues = if gpu_cores >= 30 { 5 } else if gpu_cores >= 16 { 4 } else { 3 };
    let tile_size = if gpu_cores >= 30 { 64 } else { 32 };
    VendorBackendConfig {
        gpu_cores,
        metal_queues,
        tile_size,
        simd_width: 32,
        tile_memory_bytes: 65536,
        unified_memory_bytes,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: unified_mb <= 8192,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let gpu_cores = sysctl_u64(b"hw.gpu.core_count\0").unwrap_or(8) as usize;
    requested.max(1).min(gpu_cores)
}
