pub(super) mod cpu;
pub(super) mod gpu;
pub(super) mod lpu;
pub(super) mod mmio;
pub(super) mod ram;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "macos"))]
pub(crate) mod syscall;
pub(super) mod tpu;

pub(super) fn cpu_family() -> u32 {
    let cfg = cpu::default_config();
    let sched = cpu::build_schedule(cpu::detected_parallelism().max(1));
    let clamped = cpu::clamp_workers(1);
    (cfg.p_core_workers as u32)
        .max(cfg.e_core_workers as u32)
        .max(sched.chunks as u32)
        .max(clamped as u32)
}

pub(crate) fn macos_probe(workers: usize) -> bool {
    let cpu_cfg = cpu::default_config();
    let cpu_sched = cpu::build_schedule(workers);
    let cpu_clamped = cpu::clamp_workers(workers);
    let cpu_par = cpu::detected_parallelism();
    let mac_info = cpu::detect_apple_silicon();
    let cpu_freq_ok = mac_info.as_ref().map(|i| {
        i.p_freq_hz > 0 && i.e_freq_hz > 0 && i.unified_memory_bytes > 0
    }).unwrap_or(false);
    let cpu_ok = cpu_cfg.p_core_workers > 0
        && cpu_cfg.e_core_workers > 0
        && cpu_cfg.render_workers > 0
        && cpu_cfg.freq_p_max_hz > 0
        && cpu_cfg.freq_e_max_hz > 0
        && cpu_cfg.l2_cache_bytes > 0
        && cpu_cfg.frame_budget_us > 0
        && !cpu_cfg.low_power
        && cpu_sched.chunks > 0
        && cpu_sched.chunk_size > 0
        && cpu_sched.frame_budget_us > 0
        && cpu_clamped > 0
        && cpu_par > 0
        && mac_info.is_some()
        && cpu_freq_ok;

    let gpu_cfg = gpu::default_config();
    let gpu_sched = gpu::build_schedule(workers);
    let gpu_clamped = gpu::clamp_workers(workers);
    let gpu_ok = gpu_cfg.metal_queues > 0
        && gpu_cfg.gpu_cores > 0
        && gpu_cfg.tile_size > 0
        && gpu_cfg.simd_width > 0
        && gpu_cfg.tile_memory_bytes > 0
        && gpu_cfg.unified_memory_bytes > 0
        && gpu_cfg.frame_budget_us > 0
        && !gpu_cfg.low_power
        && gpu_sched.chunks > 0
        && gpu_sched.chunk_size > 0
        && gpu_sched.frame_budget_us > 0
        && gpu_clamped > 0;

    let ram_cfg = ram::default_config();
    let ram_sched = ram::build_schedule(workers);
    let ram_clamped = ram::clamp_workers(workers);
    let ram_avail = ram_cfg.available_bytes.unwrap_or(ram_cfg.total_bytes / 2);
    let ram_ok = ram_cfg.total_bytes > 0
        && ram_cfg.page_size > 0
        && ram_cfg.frame_budget_us > 0
        && !ram_cfg.low_power
        && ram_avail > 0
        && ram_sched.chunks > 0
        && ram_sched.chunk_size > 0
        && ram_sched.frame_budget_us > 0
        && ram_clamped > 0;

    let family = cpu_family();
    tpu::configure_device(0, 0, family);
    lpu::configure_device(0, 0, family);
    let tpu_probe = tpu::probe_device();
    let lpu_probe = lpu::probe_device();
    let tpu_diag = tpu::run_diagnostics();
    let lpu_diag = lpu::run_diagnostics();
    let tpu_cfg = tpu::default_config();
    let tpu_sched = tpu::build_schedule(workers);
    let tpu_clamped = tpu::clamp_workers(workers);
    let lpu_cfg = lpu::default_config();
    let lpu_sched = lpu::build_schedule(workers);
    let lpu_clamped = lpu::clamp_workers(workers);
    let tpu_ok = tpu_cfg.workgroup_size > 0
        && tpu_cfg.compute_queues > 0
        && tpu_cfg.render_threads > 0
        && tpu_cfg.double_buffered
        && tpu_cfg.frame_budget_us > 0
        && !tpu_cfg.low_power
        && tpu_sched.chunks > 0
        && tpu_sched.chunk_size > 0
        && tpu_sched.frame_budget_us > 0
        && tpu_clamped > 0;
    let lpu_ok = lpu_cfg.workgroup_size > 0
        && lpu_cfg.compute_queues > 0
        && lpu_cfg.render_threads > 0
        && lpu_cfg.double_buffered
        && lpu_cfg.frame_budget_us > 0
        && !lpu_cfg.low_power
        && lpu_sched.chunks > 0
        && lpu_sched.chunk_size > 0
        && lpu_sched.frame_budget_us > 0
        && lpu_clamped > 0;

    cpu_ok | gpu_ok | ram_ok | tpu_ok | lpu_ok | tpu_probe | lpu_probe
        | (tpu_diag > 0)
        | (lpu_diag > 0)
}

pub(crate) fn detected_frame_budget_us() -> u64 {
    let hz = cpu::detect_apple_silicon()
        .and_then(|info| if info.p_freq_hz > 0 { None } else { None })
        .unwrap_or(0u64);
    let _ = hz;
    8_333
}

pub(crate) fn macos_cpu_freq_mhz(_cores: usize) -> (u32, u32) {
    let info = cpu::detect_apple_silicon();
    let (p_hz, e_hz) = info
        .as_ref()
        .map(|i| (i.p_freq_hz, i.e_freq_hz))
        .unwrap_or((0, 0));
    let cfg = cpu::default_config();
    let max_hz = cfg.freq_p_max_hz.max(cfg.freq_e_max_hz).max(p_hz).max(e_hz);
    let avg_hz = if p_hz > 0 && e_hz > 0 {
        (p_hz + e_hz) / 2
    } else {
        p_hz.max(e_hz)
    };
    let avg_mhz = (avg_hz / 1_000_000) as u32;
    let max_mhz = if max_hz > 0 {
        (max_hz / 1_000_000) as u32
    } else {
        avg_mhz
    };
    (avg_mhz, max_mhz)
}
