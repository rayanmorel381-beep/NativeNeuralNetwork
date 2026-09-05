use super::types::ResourceSnapshot;

pub(crate) fn snapshot(
    cpu_usage: f32,
    ram_total: usize,
    ram_available: usize,
    gpu_dispatches: usize,
    tpu_dispatches: usize,
    lpu_dispatches: usize,
) -> ResourceSnapshot {
    ResourceSnapshot {
        cpu_usage,
        ram_total,
        ram_available,
        gpu_dispatches,
        tpu_dispatches,
        lpu_dispatches,
    }
}
