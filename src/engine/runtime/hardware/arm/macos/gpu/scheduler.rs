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
    let cfg = super::backend::default_backend_config();
    let tile = cfg.tile_size.max(32);
    let queues = cfg.metal_queues.max(1);
    let raw = work_items.div_ceil(queues * cfg.gpu_cores.max(1));
    let aligned = raw.max(1).div_ceil(tile);
    aligned.max(tile)
}

pub(crate) fn build_schedule(work_items: usize) -> VendorSchedule {
    let chunk_size = recommended_chunk_size(work_items);
    let chunks = if work_items == 0 { 1 } else { work_items.div_ceil(chunk_size) };
    VendorSchedule { chunks, chunk_size, frame_budget_us: crate::engine::runtime::hardware::arch::detected_frame_budget_us() }
}
