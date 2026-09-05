use core::ffi::c_void;

#[repr(C)]
struct MemoryStatusEx {
    dw_length: u32,
    dw_memory_load: u32,
    ull_total_phys: u64,
    ull_avail_phys: u64,
    ull_total_page_file: u64,
    ull_avail_page_file: u64,
    ull_total_virtual: u64,
    ull_avail_virtual: u64,
    ull_avail_extended_virtual: u64,
}

#[repr(C)]
struct SystemInfo {
    w_processor_architecture: u16,
    w_reserved: u16,
    dw_page_size: u32,
    lp_minimum_application_address: *mut c_void,
    lp_maximum_application_address: *mut c_void,
    dw_active_processor_mask: usize,
    dw_number_of_processors: u32,
    dw_processor_type: u32,
    dw_allocation_granularity: u32,
    w_processor_level: u16,
    w_processor_revision: u16,
}

unsafe extern "system" {
    fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    fn GetSystemInfo(lp_system_info: *mut SystemInfo);
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
    let mut status = MemoryStatusEx {
        dw_length: core::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    let mut sys_info = SystemInfo {
        w_processor_architecture: 0,
        w_reserved: 0,
        dw_page_size: 4096,
        lp_minimum_application_address: core::ptr::null_mut(),
        lp_maximum_application_address: core::ptr::null_mut(),
        dw_active_processor_mask: 0,
        dw_number_of_processors: 0,
        dw_processor_type: 0,
        dw_allocation_granularity: 0,
        w_processor_level: 0,
        w_processor_revision: 0,
    };

    unsafe { GetSystemInfo(&mut sys_info) };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) } != 0;

    VendorBackendConfig {
        page_size: sys_info.dw_page_size as usize,
        total_bytes: if ok { status.ull_total_phys } else { 0 },
        available_bytes: if ok { Some(status.ull_avail_phys) } else { None },
        frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us(),
        low_power: false,
    }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    requested.max(1)
}
