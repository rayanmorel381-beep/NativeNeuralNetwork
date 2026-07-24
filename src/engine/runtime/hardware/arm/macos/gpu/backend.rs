fn infer_gpu_cores(p_cores: usize, mem_bytes: u64) -> usize {
    let mem_gb = mem_bytes / (1024 * 1024 * 1024);
    match (p_cores, mem_gb) {
        (p, m) if p >= 10 && m >= 64 => 40,
        (p, m) if p >= 8  && m >= 32 => 32,
        (p, _) if p >= 8             => 16,
        (p, m) if p >= 6  && m >= 32 => 19,
        (p, _) if p >= 6             => 16,
        (_, m) if m >= 16            => 10,
        _                            => 8,
    }
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
    let p_cores = super::sysctl_u64(b"hw.perflevel0.physicalcpu\0")
        .filter(|&n| n > 0)
        .map(|n| n as usize)
        .unwrap_or(4);
    let unified_memory_bytes = super::sysctl_u64(b"hw.memsize\0").unwrap_or(0);
    let gpu_cores = infer_gpu_cores(p_cores, unified_memory_bytes);
    let unified_mb = unified_memory_bytes / (1024 * 1024);
    let metal_queues = if gpu_cores >= 30 { 5 } else if gpu_cores >= 16 { 4 } else { 3 };
    let tile_size = if gpu_cores >= 30 { 64 } else { 32 };
    let tile_memory_bytes = if gpu_cores >= 16 { 131_072 } else { 65_536 };
    VendorBackendConfig {
        gpu_cores,
        metal_queues,
        tile_size,
        simd_width: 32,
        tile_memory_bytes,
        unified_memory_bytes,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: unified_mb <= 8192,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    let cfg = default_backend_config();
    requested.max(1).min(cfg.gpu_cores)
}
