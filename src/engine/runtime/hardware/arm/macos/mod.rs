pub(crate) mod cpu;
pub(crate) mod gpu;
pub(crate) mod lpu;
pub(super) mod mmio;
pub(crate) mod ram;
#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_os = "macos"))]
pub(crate) mod syscall;
pub(crate) mod tpu;

#[cfg(target_os = "macos")]
pub(crate) fn macos_probe(workers: usize) -> bool {
    workers >= 1 && logical_cores() > 0 && unified_memory_bytes() > 0
}

pub(crate) fn sysctl_string(name: &[u8], buf: &mut [u8]) -> Option<usize> {
    let mut len = buf.len();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr(),
            &mut len,
            core::ptr::null(),
            0,
        )
    };
    if rc == 0 && len <= buf.len() { Some(len) } else { None }
}

unsafe extern "C" {
    fn sysctlbyname(
        name: *const u8,
        oldp: *mut u8,
        oldlenp: *mut usize,
        newp: *const u8,
        newlen: usize,
    ) -> i32;
    fn getpagesize() -> i32;
}

pub(crate) fn sysctl_u64(name: &[u8]) -> Option<u64> {
    let mut val: u64 = 0;
    let mut len = core::mem::size_of::<u64>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u64 as *mut u8,
            &mut len,
            core::ptr::null(),
            0,
        )
    };
    if rc == 0 && len == core::mem::size_of::<u64>() { Some(val) } else { None }
}

pub(crate) fn sysctl_u32(name: &[u8]) -> Option<u32> {
    let mut val: u32 = 0;
    let mut len = core::mem::size_of::<u32>();
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut val as *mut u32 as *mut u8,
            &mut len,
            core::ptr::null(),
            0,
        )
    };
    if rc == 0 && len == core::mem::size_of::<u32>() { Some(val) } else { None }
}

pub(crate) fn host_page_size() -> usize {
    let p = unsafe { getpagesize() };
    if p > 0 { p as usize } else { 16384 }
}

pub(crate) fn logical_cores() -> usize {
    sysctl_u64(b"hw.logicalcpu\0")
        .filter(|&n| n > 0)
        .map(|n| n as usize)
        .unwrap_or(4)
}

pub(crate) fn p_core_count() -> usize {
    sysctl_u64(b"hw.perflevel0.physicalcpu\0")
        .filter(|&n| n > 0)
        .map(|n| n as usize)
        .unwrap_or(4)
}

pub(crate) fn e_core_count() -> usize {
    sysctl_u64(b"hw.perflevel1.physicalcpu\0")
        .map(|n| n as usize)
        .unwrap_or(0)
}

pub(crate) fn unified_memory_bytes() -> u64 {
    sysctl_u64(b"hw.memsize\0").unwrap_or(0)
}

pub(crate) fn cpu_family() -> u32 {
    sysctl_u32(b"hw.cpufamily\0").unwrap_or(0)
}

pub(crate) fn detected_frame_budget_us() -> u64 {
    if let Some(hz) = sysctl_u64(b"hw.displayrefreshrate\0") {
        if hz > 0 { return 1_000_000u64 / hz; }
    }
    8_333
}

pub(crate) fn macos_cpu_freq_mhz(_cores: usize) -> (u32, u32) {
    let p_hz = sysctl_u64(b"hw.perflevel0.cpufrequency_max\0")
        .or_else(|| sysctl_u64(b"hw.cpufrequency_max\0"))
        .unwrap_or(0);
    let e_hz = sysctl_u64(b"hw.perflevel1.cpufrequency_max\0").unwrap_or(0);
    let cur_hz = sysctl_u64(b"hw.cpufrequency\0").unwrap_or(0);
    let max_hz = p_hz.max(e_hz).max(cur_hz);
    let avg_hz = if cur_hz > 0 {
        cur_hz
    } else if p_hz > 0 && e_hz > 0 {
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
