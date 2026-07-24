use crate::observability::profiler::OpCounter;

pub fn has_recorded_work(counter: &OpCounter) -> bool {
    counter.total_ops() > 0 || counter.total_memory_bytes() > 0
}

pub fn is_memory_heavy(counter: &OpCounter) -> bool {
    counter.total_memory_bytes() > counter.total_ops()
}
