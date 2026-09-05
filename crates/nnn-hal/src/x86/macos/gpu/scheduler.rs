unsafe extern "C" {
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
        sysctlbyname(
            name.as_ptr(),
            &mut out as *mut u64 as *mut u8,
            &mut len,
            core::ptr::null(),
            0,
        )
    };
    if ret == 0 { Some(out) } else { None }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorSchedule {
    pub(crate) chunks: usize,
    pub(crate) chunk_size: usize,
    pub(crate) frame_budget_us: u64,
}

pub(crate) fn recommended_chunk_size(work_items: usize) -> usize {
    if work_items == 0 {
        return 32;
    }
    let gpu_cores = sysctl_u64(b"hw.gpu.core_count\0").unwrap_or(8) as usize;
    let tile = if gpu_cores >= 30 { 64 } else { 32 };
    let queues = if gpu_cores >= 30 { 5 } else if gpu_cores >= 16 { 4 } else { 3 };
    let raw = work_items.div_ceil(queues * gpu_cores);
    let aligned = raw.max(1).div_ceil(tile);
    aligned.max(tile)
}

pub(crate) fn build_schedule(work_items: usize) -> VendorSchedule {
    let chunk_size = recommended_chunk_size(work_items);
    let chunks = if work_items == 0 { 1 } else { work_items.div_ceil(chunk_size) };
    VendorSchedule { chunks, chunk_size, frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us() }
}
