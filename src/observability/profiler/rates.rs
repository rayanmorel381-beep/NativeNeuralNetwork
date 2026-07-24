use crate::observability::profiler::OpCounter;

pub fn ops_per_second(counter: &OpCounter, elapsed_seconds: f32) -> f32 {
    if elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return 0.0;
    }
    counter.total_ops() as f32 / elapsed_seconds
}

pub fn bytes_per_second(counter: &OpCounter, elapsed_seconds: f32) -> f32 {
    if elapsed_seconds <= 0.0 || !elapsed_seconds.is_finite() {
        return 0.0;
    }
    counter.total_memory_bytes() as f32 / elapsed_seconds
}

pub fn arithmetic_intensity(counter: &OpCounter) -> f32 {
    let bytes = counter.total_memory_bytes();
    if bytes == 0 {
        return 0.0;
    }
    counter.total_ops() as f32 / bytes as f32
}
