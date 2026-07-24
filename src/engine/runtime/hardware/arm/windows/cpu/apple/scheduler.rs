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
    let total = crate::engine::runtime::hardware::arm::detected_parallelism();
    let p_cores = (total / 2).max(1);
    let raw = work_items / p_cores;
    raw.next_power_of_two().max(4)
}

pub(crate) fn build_schedule(work_items: usize) -> VendorSchedule {
    let chunk_size = recommended_chunk_size(work_items);
    let chunks = if work_items == 0 { 1 } else { work_items.div_ceil(chunk_size) };
    VendorSchedule { chunks, chunk_size, frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us() }
}
