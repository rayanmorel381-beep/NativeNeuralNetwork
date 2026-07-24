mod agent_cache;
mod budget;
mod estimate;
mod flops;
pub(crate) mod hardware;
mod no_alloc;
mod parallel;
mod profile;
mod thread_pool;
pub(crate) use agent_cache::{
    cache_ensure, cache_full, cache_release, cache_replica_count, cache_replica_slice,
    cache_reset_used, cache_store, compact_f32, compact_f64,
};

pub(crate) use parallel::{parallel_for, SyncMutPtr, set_parallel_executor, reset_parallel_executor};
pub(crate) use thread_pool::thread_parallel_for;
pub(crate) use thread_pool::parallel_memcpy;
pub(crate) use thread_pool::install_as_parallel_executor;
pub(crate) use hardware::{axpy, dot, dot_f32, l2_cache_bytes};

pub use crate::base::precision::Precision;
pub(crate) use budget::{check_runtime_budget, fit_from_estimate, BudgetFit};pub(crate) use estimate::{estimate_runtime_memory, RuntimeError, RuntimeEstimate};
pub(crate) use flops::{
    estimate_runtime_flops, estimate_tokens_per_second, RuntimeFlopsEstimate, ThroughputEstimate,
};
pub use no_alloc::FixedSliceVec;
pub use no_alloc::FixedWriter;
pub use profile::RuntimeProfile;
pub(crate) use hardware::{
    derive_runtime_plan,
    detect_hardware_profile,
    SwapBatch, adjusted_samples_per_cycle,
    ensure_hardware_init,
    gpu_matmul,
    gpu_gemm,
};
pub(crate) use hardware::ConsumptionGuard;

pub fn set_precision(precision: Precision) {
    crate::base::precision::set_precision(precision);
}

pub fn gpu_device_probe() -> Option<(u32, u32, u32)> {
    hardware::gpu_device_probe()
}

pub fn gpu_gem_selftest(elements: usize) -> Option<(u32, usize, bool, bool)> {
    hardware::gpu_gem_selftest(elements)
}

pub fn gpu_compute_selftest(elements: usize) -> Option<(usize, f32, f32, bool)> {
    hardware::gpu_compute_selftest(elements)
}

pub struct GpuMatmulProfileArgs<'a> {
    pub src: &'a [u8],
    pub dst: &'a mut [u8],
    pub batch_size: usize,
    pub stride: usize,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: &'a [u8],
    pub biases: &'a [u8],
    pub activation: u32,
}

pub fn gpu_matmul_profile(args: GpuMatmulProfileArgs<'_>) -> Option<(u64, u64, u64, u64, u64)> {
    let mut gpu_probe_buf = [0u8; 4];
    if let Some(mut gt) = crate::base::tensor::GpuTensor::from_raw(
        gpu_probe_buf.as_mut_ptr(), 4, [1, 1, 1, 1, 1]
    ) {
        let _ = gt.numel_f32();
        let _ = gt.as_bytes().len();
        let _ = gt.as_bytes_mut().len();
        let _ = gt.shape[4];
        let _ = gpu_tensor_profile(&gt);
    }
    hardware::gpu_matmul_profile(
        args.src,
        args.dst,
        args.batch_size,
        args.stride,
        args.in_size,
        args.out_size,
        args.weights,
        args.biases,
        args.activation,
    )
}

pub fn gpu_tensor_profile(tensor: &crate::base::tensor::GpuTensor) -> Option<(u64, u64, u64, u64, u64)> {
    let n = tensor.numel_f32();
    if n == 0 { return None; }
    let bytes = tensor.as_bytes();
    let mut dummy_out = [0u8; 4];
    hardware::gpu_matmul_profile(
        bytes,
        &mut dummy_out,
        1,
        n,
        n,
        1,
        &[],
        &[],
        0,
    )
}



