#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) mod linux;
#[cfg(target_os = "macos")]
pub(crate) mod macos;
#[cfg(target_os = "windows")]
pub(crate) mod windows;
pub(crate) mod os;
#[cfg(target_arch = "x86_64")]
pub(crate) mod simd;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) use os::{linux_probe, linux_hardware_data, linux_cpu_cores};
#[cfg(target_os = "windows")]
pub(crate) use os::{windows_probe, windows_hardware_data, windows_cpu_cores};
#[cfg(target_os = "macos")]
pub(crate) use macos::macos_probe;

pub(crate) fn detected_parallelism() -> usize {
    let count: u32;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "xchg {tmp:r}, rbx",
            "cpuid",
            "mov {count:e}, ebx",
            "xchg {tmp:r}, rbx",
            tmp = out(reg) _,
            count = out(reg) count,
            inout("eax") 0x0Bu32 => _,
            inout("ecx") 1u32 => _,
            out("edx") _,
            options(nostack, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        let mut ebx_out: u32;
        core::arch::asm!(
            "xchg {tmp:e}, ebx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "xchg {tmp:e}, ebx",
            tmp = out(reg) _,
            ebx_out = out(reg) ebx_out,
            inout("eax") 0x0Bu32 => _,
            inout("ecx") 1u32 => _,
            out("edx") _,
            options(nostack, preserves_flags)
        );
        count = ebx_out;
    }
    let cpuid = count as usize;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let live = linux::live_cpu_count();
        cpuid.max(live).max(1)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        cpuid.max(1)
    }
}

pub(crate) fn detected_frame_budget_us() -> u64 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    { linux::detected_frame_budget_us() }
    #[cfg(target_os = "macos")]
    { macos::detected_frame_budget_us() }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    { 8_333 }
}

pub(crate) fn process_identifier() -> Option<&'static str> {
    let mut eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "xchg {tmp:r}, rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "xchg {tmp:r}, rbx",
            tmp = out(reg) _,
            ebx_out = out(reg) ebx,
            inout("eax") 0u32 => eax,
            out("ecx") ecx,
            out("edx") edx,
            options(nostack, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86")]
    unsafe {
        core::arch::asm!(
            "xchg {tmp:e}, ebx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "xchg {tmp:e}, ebx",
            tmp = out(reg) _,
            ebx_out = out(reg) ebx,
            inout("eax") 0u32 => eax,
            out("ecx") ecx,
            out("edx") edx,
            options(nostack, preserves_flags)
        );
    }
    let _ = eax;
    let bytes: [u8; 12] = unsafe { core::mem::transmute([ebx, edx, ecx]) };
    if &bytes == b"AuthenticAMD" {
        return Some("amd");
    }
    if &bytes == b"GenuineIntel" {
        return Some("intel");
    }
    Some("intel")
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_cpu_cores(workers: usize) -> usize {
    workers.max(detected_parallelism()).max(1)
}

#[cfg(target_os = "macos")]
pub(crate) fn macos_hardware_data(workers: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
    let n = workers.max(1);
    (n, n, false, false, false, 0, 0, 0)
}

#[cfg(any(target_os = "linux", target_os = "android", target_os = "windows"))]
pub(crate) fn contains_ascii_nocase(haystack: &str, needle: &str) -> bool {
    crate::arch::contains_ascii_nocase(haystack, needle)
}
