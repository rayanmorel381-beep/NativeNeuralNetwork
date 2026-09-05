#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VendorSchedule {
    pub(crate) chunks: usize,
    pub(crate) chunk_size: usize,
    pub(crate) frame_budget_us: u64,
}

pub(crate) fn recommended_chunk_size(work_items: usize) -> usize {
    if work_items == 0 {
        return 4;
    }
    let p_cores = super::sysctl_u64(b"hw.perflevel0.physicalcpu\0")
        .filter(|&n| n > 0)
        .map(|n| n as usize)
        .unwrap_or(4);
    let render_workers = p_cores.saturating_sub(1).max(1);
    let raw = work_items.div_ceil(render_workers);
    raw.next_power_of_two().max(4)
}

pub(crate) fn build_schedule(work_items: usize) -> VendorSchedule {
    let chunk_size = recommended_chunk_size(work_items);
    let chunks = if work_items == 0 { 1 } else { work_items.div_ceil(chunk_size) };
    VendorSchedule { chunks, chunk_size, frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us() }
}
