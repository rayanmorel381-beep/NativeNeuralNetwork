#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn linux_probe(workers: usize) -> bool {
    super::linux::probe_cpu(workers)
        | super::linux::probe_gpu(workers)
        | super::linux::probe_ram(workers)
        | super::linux::probe_tpu(workers)
        | super::linux::probe_lpu(workers)
}

#[cfg(target_os = "windows")]
pub(crate) fn windows_probe(workers: usize) -> bool {
    super::windows::probe_cpu(workers)
        | super::windows::probe_gpu(workers)
        | super::windows::probe_ram(workers)
        | super::windows::probe_tpu(workers)
        | super::windows::probe_lpu(workers)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn linux_cpu_cores(workers: usize) -> usize {
    let live = super::linux::live_cpu_count();
    let cpu_cfg = super::linux::cpu::vendor::default_config();
    let cpu_sched = super::linux::cpu::vendor::build_schedule(workers);
    let cpu_clamped = super::linux::cpu::vendor::clamp_workers(workers);
    let cpu_base = cpu_cfg.worker_hint.max(cpu_cfg.render_workers).max(1);
    let baseline = if cpu_cfg.low_power || cpu_cfg.frame_budget_us == 0 {
        cpu_base.min(cpu_cfg.render_workers.max(1))
    } else {
        let sched_hint = if cpu_sched.chunk_size > 0 && cpu_sched.frame_budget_us > 0 {
            cpu_sched.chunks.max(cpu_clamped)
        } else {
            1
        };
        cpu_base.max(sched_hint).max(workers)
    };
    baseline.max(live).max(workers).max(1)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) fn linux_hardware_data(workers: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
    let (live_total, live_avail, live_page) = super::linux::live_ram();
    let ram_cfg = super::linux::ram::vendor::default_config();
    let ram_sched = super::linux::ram::vendor::build_schedule(workers);
    let ram_clamped = super::linux::ram::vendor::clamp_workers(workers);
    let cfg_total = ram_cfg.total_bytes;
    let cfg_avail = ram_cfg.available_bytes.unwrap_or(cfg_total / 2);
    let total_bytes = if live_total > 0 { live_total } else { cfg_total };
    let avail_bytes = if live_avail > 0 { live_avail } else { cfg_avail };
    let ram_total = total_bytes as usize;
    let ram_sched_ok = ram_sched.chunks > 0 && ram_sched.frame_budget_us > 0;
    let ram_page = if ram_cfg.page_size > 0 { ram_cfg.page_size } else { live_page };
    let ram_page_ok = ram_page > 0 && ram_cfg.frame_budget_us > 0 && !ram_cfg.low_power;
    let ram_available = if ram_sched_ok && ram_page_ok {
        (avail_bytes as usize).min(ram_total).max(ram_sched.chunk_size.min(ram_clamped + 1))
    } else {
        avail_bytes as usize
    };

    let gpu_present = super::linux::gpu_device_present();
    let gpu_cfg = super::linux::gpu::vendor::default_config();
    let gpu_sched = super::linux::gpu::vendor::build_schedule(workers);
    let gpu_clamped = super::linux::gpu::vendor::clamp_workers(workers);
    let gpu_active = gpu_present
        && gpu_cfg.workgroup_size > 0
        && gpu_cfg.compute_queues > 0
        && gpu_sched.chunks > 0
        && gpu_cfg.render_threads > 0
        && gpu_cfg.frame_budget_us > 0
        && !gpu_cfg.low_power
        && gpu_cfg.double_buffered
        && gpu_clamped > 0
        && gpu_sched.chunk_size > 0
        && gpu_sched.frame_budget_us > 0;

    let tpu_cfg = super::linux::tpu::vendor::default_config();
    let tpu_sched = super::linux::tpu::vendor::build_schedule(workers);
    let tpu_clamped = super::linux::tpu::vendor::clamp_workers(workers);
    let tpu_active = tpu_cfg.workgroup_size > 0
        && tpu_cfg.compute_queues > 0
        && tpu_sched.chunks > 0
        && tpu_cfg.render_threads > 0
        && tpu_cfg.frame_budget_us > 0
        && !tpu_cfg.low_power
        && tpu_cfg.double_buffered
        && tpu_clamped > 0
        && tpu_sched.chunk_size > 0
        && tpu_sched.frame_budget_us > 0;

    let lpu_cfg = super::linux::lpu::vendor::default_config();
    let lpu_sched = super::linux::lpu::vendor::build_schedule(workers);
    let lpu_clamped = super::linux::lpu::vendor::clamp_workers(workers);
    let lpu_active = lpu_cfg.workgroup_size > 0
        && lpu_cfg.compute_queues > 0
        && lpu_sched.chunks > 0
        && lpu_cfg.render_threads > 0
        && lpu_cfg.frame_budget_us > 0
        && !lpu_cfg.low_power
        && lpu_cfg.double_buffered
        && lpu_clamped > 0
        && lpu_sched.chunk_size > 0
        && lpu_sched.frame_budget_us > 0;

    let gpu_dispatches = if gpu_active { gpu_sched.chunks } else { 0 };
    let tpu_dispatches = if tpu_active { tpu_sched.chunks } else { 0 };
    let lpu_dispatches = if lpu_active { lpu_sched.chunks } else { 0 };
    (ram_total.max(1), ram_available.max(1), gpu_active, tpu_active, lpu_active, gpu_dispatches, tpu_dispatches, lpu_dispatches)
}

#[cfg(target_os = "windows")]
pub(crate) fn windows_cpu_cores(workers: usize) -> usize {
    let cpu_cfg = super::windows::cpu::vendor::default_config();
    let cpu_sched = super::windows::cpu::vendor::build_schedule(workers);
    let cpu_clamped = super::windows::cpu::vendor::clamp_workers(workers);
    let cpu_base = cpu_cfg.worker_hint.max(cpu_cfg.render_workers).max(1);
    if cpu_cfg.low_power || cpu_cfg.frame_budget_us == 0 {
        cpu_base.min(cpu_cfg.render_workers.max(1))
    } else {
        let sched_hint = if cpu_sched.chunk_size > 0 && cpu_sched.frame_budget_us > 0 {
            cpu_sched.chunks.max(cpu_clamped)
        } else {
            1
        };
        cpu_base.max(sched_hint).max(workers)
    }
    .max(1)
}

#[cfg(target_os = "windows")]
pub(crate) fn windows_hardware_data(workers: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
    let ram_cfg = super::windows::ram::vendor::default_config();
    let ram_avail = ram_cfg.available_bytes.unwrap_or(ram_cfg.total_bytes / 2);
    let ram_total = ram_cfg.total_bytes as usize;
    let _ = workers;
    (ram_total.max(1), (ram_avail as usize).max(1), false, false, false, 0, 0, 0)
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_cpu_cores(workers: usize) -> usize {
    let cores = super::macos::logical_cores();
    cores.max(workers).max(1)
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_hardware_data(workers: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
    let _ = workers;
    let ram = super::macos::unified_memory_bytes() as usize;
    (ram.max(1), (ram / 2).max(1), false, false, false, 0, 0, 0)
}
