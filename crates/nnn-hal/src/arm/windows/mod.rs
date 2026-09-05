pub(super) mod cpu;
pub(super) mod gpu;
pub(super) mod lpu;
pub(super) mod ram;
pub(super) mod tpu;

#[cfg(target_os = "windows")]
mod freq {
    use core::ffi::c_void;

    type Hkey = *mut c_void;
    const HKEY_LOCAL_MACHINE: Hkey = 0x80000002_usize as Hkey;
    const KEY_READ: u32 = 0x20019;

    unsafe extern "system" {
        fn RegOpenKeyExW(key: Hkey, sub_key: *const u16, options: u32, desired: u32, result: *mut Hkey) -> i32;
        fn RegQueryValueExW(key: Hkey, value_name: *const u16, reserved: *mut u32, reg_type: *mut u32, data: *mut u8, data_len: *mut u32) -> i32;
        fn RegCloseKey(key: Hkey) -> i32;
    }

    fn to_wide(s: &str, buf: &mut [u16; 128]) {
        let mut len = 0;
        for c in s.encode_utf16() {
            if len + 1 >= 128 { break; }
            buf[len] = c;
            len += 1;
        }
        buf[len] = 0;
    }

    fn read_mhz_dword(cpu: usize) -> u32 {
        let mut key_str = [0u8; 64];
        let prefix = b"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\";
        let mut w = 0usize;
        for &b in prefix { key_str[w] = b; w += 1; }
        if cpu == 0 {
            key_str[w] = b'0';
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
                key_str[w] = digits[d];
                w += 1;
            }
        }
        let key_ascii = match core::str::from_utf8(&key_str[..w]) {
            Ok(s) => s,
            Err(_) => return 0,
        };
        let mut key_wide = [0u16; 128];
        let mut val_wide = [0u16; 128];
        to_wide(key_ascii, &mut key_wide);
        to_wide("~MHz", &mut val_wide);
        let mut hkey: Hkey = core::ptr::null_mut();
        let open = unsafe { RegOpenKeyExW(HKEY_LOCAL_MACHINE, key_wide.as_ptr(), 0, KEY_READ, &mut hkey) };
        if open != 0 { return 0; }
        let mut value: u32 = 0;
        let mut len: u32 = 4;
        let mut reg_type: u32 = 0;
        let res = unsafe {
            RegQueryValueExW(
                hkey,
                val_wide.as_ptr(),
                core::ptr::null_mut(),
                &mut reg_type,
                &mut value as *mut u32 as *mut u8,
                &mut len,
            )
        };
        unsafe { RegCloseKey(hkey) };
        if res == 0 { value } else { 0 }
    }

    pub(crate) fn windows_cpu_freq_mhz(cores: usize) -> (u32, u32) {
        let n = cores.clamp(1, 256);
        let mut sum = 0u64;
        let mut counted = 0u64;
        let mut max = 0u32;
        for c in 0..n {
            let mhz = read_mhz_dword(c);
            if mhz > 0 {
                sum += mhz as u64;
                counted += 1;
                if mhz > max { max = mhz; }
            }
        }
        let avg = if counted > 0 { (sum / counted) as u32 } else { 0 };
        (avg, if max > 0 { max } else { avg })
    }
}

#[cfg(target_os = "windows")]
pub(crate) use freq::windows_cpu_freq_mhz;

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
