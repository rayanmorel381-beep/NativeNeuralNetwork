pub(super) mod cpu;
pub(super) mod gpu;
pub(super) mod lpu;
pub(super) mod ram;
#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
pub(crate) mod syscall;
pub(super) mod tpu;

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn live_cpu_count() -> usize {
    let mut mask = [0usize; 16];
    let rc = syscall::sys_sched_getaffinity(0, &mut mask);
    if rc <= 0 {
        return cpu_count_proc().unwrap_or(1);
    }
    let mut count = 0usize;
    for w in mask.iter() {
        count = count.saturating_add(w.count_ones() as usize);
    }
    count.max(1)
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn live_cpu_count() -> usize { 1 }

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn cpu_count_proc() -> Option<usize> {
    let path = b"/sys/devices/system/cpu/online\0";
    let mut buf = [0u8; 256];
    let n = read_small_file(path, &mut buf)?;
    Some(parse_cpu_list(&buf[..n]))
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn parse_cpu_list(bytes: &[u8]) -> usize {
    let mut count = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let mut start = 0usize;
        let mut has = false;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            start = start.saturating_mul(10).saturating_add((bytes[i] - b'0') as usize);
            has = true;
            i += 1;
        }
        if !has { i += 1; continue; }
        if i < bytes.len() && bytes[i] == b'-' {
            i += 1;
            let mut end = 0usize;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                end = end.saturating_mul(10).saturating_add((bytes[i] - b'0') as usize);
                i += 1;
            }
            count = count.saturating_add(end.saturating_sub(start).saturating_add(1));
        } else {
            count = count.saturating_add(1);
        }
        while i < bytes.len() && (bytes[i] == b',' || bytes[i] == b' ' || bytes[i] == b'\n') { i += 1; }
    }
    count
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn live_ram() -> (u64, u64, usize) {
    let (mem_total_kb, mem_available_kb) = parse_meminfo();
    if mem_total_kb > 0 {
        let available = if mem_available_kb > 0 {
            mem_available_kb.saturating_mul(1024)
        } else {
            sysinfo_available_bytes()
        };
        return (mem_total_kb.saturating_mul(1024), available, 4096);
    }
    let mut info = new_sysinfo();
    let rc = syscall::sys_sysinfo(&mut info);
    if rc < 0 || info.mem_unit == 0 { return (0, 0, 4096); }
    let unit = info.mem_unit as u64;
    let total = (info.totalram as u64).saturating_mul(unit);
    let free = (info.freeram as u64).saturating_mul(unit);
    let buffers = (info.bufferram as u64).saturating_mul(unit);
    (total, free.saturating_add(buffers), 4096)
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn live_ram() -> (u64, u64, usize) { (0, 0, 4096) }

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn new_sysinfo() -> syscall::SysInfo {
    syscall::SysInfo {
        uptime: 0, loads: [0; 3], totalram: 0, freeram: 0, sharedram: 0,
        bufferram: 0, totalswap: 0, freeswap: 0, procs: 0, _pad1: 0,
        totalhigh: 0, freehigh: 0, mem_unit: 0, _pad2: 0,
    }
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn sysinfo_available_bytes() -> u64 {
    let mut info = new_sysinfo();
    let rc = syscall::sys_sysinfo(&mut info);
    if rc < 0 || info.mem_unit == 0 { return 0; }
    let unit = info.mem_unit as u64;
    let free = (info.freeram as u64).saturating_mul(unit);
    let buffers = (info.bufferram as u64).saturating_mul(unit);
    free.saturating_add(buffers)
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn parse_meminfo() -> (u64, u64) {
    let path = b"/proc/meminfo\0";
    let mut buf = [0u8; 4096];
    let n = match read_small_file(path, &mut buf) { Some(v) => v, None => return (0, 0) };
    let bytes = &buf[..n];
    (find_kb_field(bytes, b"MemTotal:"), find_kb_field(bytes, b"MemAvailable:"))
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn find_kb_field(bytes: &[u8], key: &[u8]) -> u64 {
    let mut i = 0usize;
    while i + key.len() <= bytes.len() {
        if &bytes[i..i + key.len()] == key {
            let mut j = i + key.len();
            while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\t') { j += 1; }
            let mut value: u64 = 0;
            let mut has = false;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                value = value.saturating_mul(10).saturating_add((bytes[j] - b'0') as u64);
                has = true;
                j += 1;
            }
            return if has { value } else { 0 };
        }
        while i < bytes.len() && bytes[i] != b'\n' { i += 1; }
        i += 1;
    }
    0
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn read_small_file(path: &[u8], buf: &mut [u8]) -> Option<usize> {
    let fd = syscall::sys_open(path, syscall::O_RDONLY, 0);
    if fd < 0 { return None; }
    let mut total = 0usize;
    while total < buf.len() {
        let n = syscall::sys_read_fd(fd, &mut buf[total..]);
        if n < 0 { let _ = syscall::sys_close(fd); return None; }
        if n == 0 { break; }
        total = total.saturating_add(n as usize);
    }
    let _ = syscall::sys_close(fd);
    Some(total)
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
fn read_cpufreq_khz(cpu: usize, field: &[u8]) -> u64 {
    let mut path = [0u8; 96];
    let prefix = b"/sys/devices/system/cpu/cpu";
    let mid = b"/cpufreq/";
    let mut w = 0usize;
    for &b in prefix { path[w] = b; w += 1; }
    if cpu == 0 {
        path[w] = b'0';
        w += 1;
    } else {
        let mut digits = [0u8; 20];
        let mut d = 0usize;
        let mut n = cpu;
        while n > 0 {
            digits[d] = b'0' + (n % 10) as u8;
            d += 1;
            n /= 10;
        }
        while d > 0 {
            d -= 1;
            path[w] = digits[d];
            w += 1;
        }
    }
    for &b in mid { path[w] = b; w += 1; }
    for &b in field { path[w] = b; w += 1; }
    path[w] = 0;
    let mut buf = [0u8; 32];
    let n = match read_small_file(&path[..=w], &mut buf) {
        Some(v) => v,
        None => return 0,
    };
    let mut value = 0u64;
    let mut has = false;
    for &b in &buf[..n] {
        if b.is_ascii_digit() {
            value = value.saturating_mul(10).saturating_add((b - b'0') as u64);
            has = true;
        } else {
            break;
        }
    }
    if has { value } else { 0 }
}

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn live_cpu_freq_mhz(cores: usize) -> (u32, u32) {
    let n = cores.clamp(1, 256);
    let mut sum_khz = 0u64;
    let mut counted = 0u64;
    let mut max_khz = 0u64;
    for c in 0..n {
        let cur = read_cpufreq_khz(c, b"scaling_cur_freq");
        if cur > 0 {
            sum_khz = sum_khz.saturating_add(cur);
            counted += 1;
        }
        let cmax = read_cpufreq_khz(c, b"cpuinfo_max_freq");
        if cmax > max_khz {
            max_khz = cmax;
        }
    }
    let avg_mhz = if counted > 0 {
        (sum_khz / counted / 1000) as u32
    } else {
        0
    };
    let max_mhz = if max_khz > 0 {
        (max_khz / 1000) as u32
    } else {
        avg_mhz
    };
    (avg_mhz, max_mhz)
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn live_cpu_freq_mhz(_cores: usize) -> (u32, u32) { (0, 0) }

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_device_present() -> bool {
    const CANDIDATES: &[&[u8]] = &[
        b"/sys/class/kgsl/kgsl-3d0\0",
        b"/dev/kgsl-3d0\0",
        b"/sys/class/devfreq/gpufreq\0",
        b"/dev/mali0\0",
        b"/dev/mali1\0",
        b"/sys/class/misc/mali0\0",
        b"/dev/dri/renderD128\0",
        b"/dev/dri/renderD129\0",
        b"/dev/dri/card0\0",
    ];
    for path in CANDIDATES {
        let fd = syscall::sys_open(path, syscall::O_RDONLY, 0);
        if fd >= 0 { let _ = syscall::sys_close(fd); return true; }
    }
    false
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_device_present() -> bool { false }

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn detected_frame_budget_us() -> u64 {
    const PATHS: &[&[u8]] = &[
        b"/sys/class/graphics/fb0/modes\0",
        b"/sys/class/graphics/fb1/modes\0",
    ];
    let mut buf = [0u8; 64];
    for path in PATHS {
        if let Some(n) = read_small_file(path, &mut buf) {
            let s = &buf[..n];
            let mut i = 0usize;
            while i < s.len() && s[i] != b'-' { i += 1; }
            if i < s.len() { i += 1; }
            let mut hz: u64 = 0;
            let mut has = false;
            while i < s.len() && s[i].is_ascii_digit() {
                hz = hz.saturating_mul(10).saturating_add((s[i] - b'0') as u64);
                has = true;
                i += 1;
            }
            if has && hz > 0 { return 1_000_000u64 / hz; }
        }
    }
    8_333
}

#[cfg(not(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn detected_frame_budget_us() -> u64 { 8_333 }

pub(super) fn probe_cpu(workers: usize) -> bool {
    let cfg = cpu::vendor::default_config();
    let sched = cpu::vendor::build_schedule(workers);
    let clamped = cpu::vendor::clamp_workers(workers);
    let min_hint = match cfg.vendor {
        cpu::vendor::Vendor::Apple => cfg.render_workers.max(1),
        _ => cfg.worker_hint.max(1),
    };
    cfg.frame_budget_us > 0
        && sched.chunk_size > 0
        && sched.frame_budget_us > 0
        && (cfg.low_power || sched.chunks >= clamped)
        && min_hint > 0
}

pub(super) fn probe_gpu(workers: usize) -> bool {
    let cfg = gpu::vendor::default_config();
    let sched = gpu::vendor::build_schedule(workers);
    let clamped = gpu::vendor::clamp_workers(workers);
    let queue_hint = match cfg.vendor {
        gpu::vendor::Vendor::Amd => cfg.compute_queues * 2,
        _ => cfg.compute_queues,
    };
    cfg.workgroup_size > 0
        && cfg.render_threads > 0
        && cfg.frame_budget_us > 0
        && sched.chunk_size > 0
        && sched.frame_budget_us > 0
        && (cfg.low_power || cfg.double_buffered || sched.chunks >= clamped)
        && queue_hint > 0
}

pub(super) fn probe_ram(workers: usize) -> bool {
    let cfg = ram::vendor::default_config();
    let sched = ram::vendor::build_schedule(workers);
    let clamped = ram::vendor::clamp_workers(workers);
    let avail = cfg.available_bytes.unwrap_or(cfg.total_bytes / 2);
    let page_ok = match cfg.vendor {
        ram::vendor::Vendor::Apple => cfg.page_size >= 4096,
        _ => cfg.page_size > 0,
    };
    cfg.total_bytes > 0
        && cfg.frame_budget_us > 0
        && avail > 0
        && page_ok
        && sched.chunk_size > 0
        && sched.frame_budget_us > 0
        && (cfg.low_power || sched.chunks >= clamped)
}

pub(super) fn probe_tpu(workers: usize) -> bool {
    let cfg = tpu::vendor::default_config();
    let sched = tpu::vendor::build_schedule(workers);
    let clamped = tpu::vendor::clamp_workers(workers);
    let queue_hint = match cfg.vendor {
        tpu::vendor::Vendor::Apple => cfg.compute_queues,
        _ => cfg.workgroup_size.max(cfg.compute_queues),
    };
    cfg.render_threads > 0
        && cfg.frame_budget_us > 0
        && sched.chunk_size > 0
        && sched.frame_budget_us > 0
        && (cfg.low_power || cfg.double_buffered || sched.chunks >= clamped)
        && queue_hint > 0
}

pub(super) fn probe_lpu(workers: usize) -> bool {
    let cfg = lpu::vendor::default_config();
    let sched = lpu::vendor::build_schedule(workers);
    let clamped = lpu::vendor::clamp_workers(workers);
    let queue_hint = match cfg.vendor {
        lpu::vendor::Vendor::Apple => cfg.compute_queues,
        _ => cfg.workgroup_size.max(cfg.compute_queues),
    };
    cfg.render_threads > 0
        && cfg.frame_budget_us > 0
        && sched.chunk_size > 0
        && sched.frame_budget_us > 0
        && (cfg.low_power || cfg.double_buffered || sched.chunks >= clamped)
        && queue_hint > 0
}
