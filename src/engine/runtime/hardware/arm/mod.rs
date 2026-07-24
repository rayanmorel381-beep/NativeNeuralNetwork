#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) mod linux;
#[cfg(target_os = "macos")]
pub(crate) mod macos;
#[cfg(target_os = "windows")]
pub(crate) mod windows;
pub(crate) mod os;

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(crate) use os::{linux_probe, linux_hardware_data, linux_cpu_cores};
#[cfg(target_os = "windows")]
pub(crate) use os::{windows_probe, windows_hardware_data, windows_cpu_cores};
#[cfg(target_os = "macos")]
pub(crate) use os::{macos_cpu_cores, macos_hardware_data};
#[cfg(target_os = "macos")]
pub(crate) use macos::macos_probe;

pub(crate) fn detected_parallelism() -> usize {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        linux::live_cpu_count()
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        1
    }
}

pub(crate) fn detected_frame_budget_us() -> u64 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    { return linux::detected_frame_budget_us(); }
    #[cfg(target_os = "macos")]
    { return macos::detected_frame_budget_us(); }
    #[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
    { 8_333 }
}

pub(crate) fn process_identifier() -> Option<&'static str> {
    None
}

pub(crate) fn contains_ascii_nocase(haystack: &str, needle: &str) -> bool {
    crate::engine::runtime::hardware::arch::contains_ascii_nocase(haystack, needle)
}
