unsafe extern "C" {
    fn getpagesize() -> i32;
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
        sysctlbyname(name.as_ptr(), &mut out as *mut u64 as *mut u8, &mut len, core::ptr::null(), 0)
    };
    if ret == 0 { Some(out) } else { None }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorBackendConfig {
    pub(crate) page_size: usize,
    pub(crate) total_bytes: u64,
    pub(crate) available_bytes: Option<u64>,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

pub(crate) fn default_backend_config() -> VendorBackendConfig {
    let page = unsafe { getpagesize() };
    VendorBackendConfig {
        page_size: if page > 0 { page as usize } else { 4096 },
        total_bytes: sysctl_u64(b"hw.memsize\0").unwrap_or(0),
        available_bytes: None,
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: false,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    requested.max(1)
}
