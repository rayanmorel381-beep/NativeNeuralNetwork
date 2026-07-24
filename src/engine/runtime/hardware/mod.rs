#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod x86;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
pub mod arm;
pub mod arch;
pub mod consumption;
pub mod types;

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
use arm::linux::syscall as backend;
#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_os = "macos"))]
use arm::macos::syscall as backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
use x86::linux::syscall as backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "macos"))]
use x86::macos::syscall as backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "windows"))]
use x86::windows::syscall as backend;

pub const O_RDONLY: i32 = backend::O_RDONLY;
pub const O_WRONLY: i32 = backend::O_WRONLY;
pub const O_APPEND: i32 = backend::O_APPEND;

pub fn o_creat() -> i32 { backend::O_CREAT }
pub fn o_trunc() -> i32 { backend::O_TRUNC }

pub fn sys_open(path: &[u8], flags: i32, mode: u32) -> i64 { backend::sys_open(path, flags, mode) }
pub fn sys_close(fd: i64) -> i64 { backend::sys_close(fd) }
pub fn sys_write_fd(fd: i64, buf: &[u8]) -> i64 { backend::sys_write_fd(fd, buf) }
pub fn sys_read_fd(fd: i64, buf: &mut [u8]) -> i64 { backend::sys_read_fd(fd, buf) }
pub fn sys_mkdir(path: &[u8], mode: u32) -> i64 { backend::sys_mkdir(path, mode) }
pub fn mmap_shared_anon(size: usize) -> *mut u8 { backend::mmap_shared_anon(size) }
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub fn mmap_private_anon(size: usize) -> *mut u8 { backend::mmap_private_anon(size) }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub fn mmap_private_anon(size: usize) -> *mut u8 { backend::mmap_shared_anon(size) }
pub fn munmap(ptr: *mut u8, size: usize) { backend::munmap(ptr, size) }
pub fn monotonic_ns() -> u64 { backend::monotonic_ns() }
pub fn fork() -> i64 { backend::fork() }
pub fn waitpid(pid: i64) { backend::waitpid(pid) }
pub fn exit(code: i32) -> ! { backend::exit(code) }
pub fn set_affinity(mask: usize) { backend::set_affinity(mask) }
pub use backend::{DirEnt64, sys_getdents64};

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wait_u32(addr: *const u32, val: u32) { backend::futex_wait_u32(addr, val); }
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn futex_wake_u32(addr: *const u32, n: u32) { backend::futex_wake_u32(addr, n); }
#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn prctl_set_pdeathsig(sig: usize) { backend::prctl_set_pdeathsig(sig); }

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
pub unsafe fn spawn_thread(
    flags: usize,
    child_stack_top: usize,
    entry: extern "C" fn(usize),
    arg: usize,
    done_ptr: usize,
) -> i64 {
    backend::clone_thread(flags, child_stack_top, entry, arg, done_ptr)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub fn gpu_device_probe() -> Option<(u32, u32, u32)> { x86::linux::radeon_probe() }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub fn gpu_device_probe() -> Option<(u32, u32, u32)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub fn gpu_gem_selftest(elements: usize) -> Option<(u32, usize, bool, bool)> { x86::linux::radeon_gem_selftest(elements) }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub fn gpu_gem_selftest(_elements: usize) -> Option<(u32, usize, bool, bool)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub fn gpu_compute_selftest(elements: usize) -> Option<(usize, f32, f32, bool)> { x86::linux::radeon_gpu_compute_selftest(elements) }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub fn gpu_compute_selftest(_elements: usize) -> Option<(usize, f32, f32, bool)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul(
    src: &[u8],
    dst: &mut [u8],
    batch_size: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
    weights: &[u8],
    biases: &[u8],
    activation: u32,
) -> bool {
    x86::linux::gpu_matmul_dispatch(
        src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
    )
}
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul(
    _src: &[u8],
    _dst: &mut [u8],
    _batch_size: usize,
    _stride: usize,
    _in_size: usize,
    _out_size: usize,
    _weights: &[u8],
    _biases: &[u8],
    _activation: u32,
) -> bool {
    false
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub fn gpu_gemm(
    elem_size: usize,
    x: &[u8],
    y: &mut [u8],
    rows: usize,
    in_d: usize,
    out_d: usize,
    weights: &[u8],
) -> bool {
    x86::linux::gpu_gemm_dispatch(elem_size, x, y, rows, in_d, out_d, weights)
}
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub fn gpu_gemm(
    _elem_size: usize,
    _x: &[u8],
    _y: &mut [u8],
    _rows: usize,
    _in_d: usize,
    _out_d: usize,
    _weights: &[u8],
) -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul_profile(
    src: &[u8],
    dst: &mut [u8],
    batch_size: usize,
    stride: usize,
    in_size: usize,
    out_size: usize,
    weights: &[u8],
    biases: &[u8],
    activation: u32,
) -> Option<(u64, u64, u64, u64, u64)> {
    let started = monotonic_ns();
    let ok = gpu_matmul(src, dst, batch_size, stride, in_size, out_size, weights, biases, activation);
    let elapsed = monotonic_ns().saturating_sub(started);
    if ok { Some((0, 0, 0, 0, elapsed)) } else { None }
}

pub use types::SwapBatch;
pub use arch::{
    adjusted_samples_per_cycle,
    derive_runtime_plan, detect_hardware_profile,
    ensure_hardware_init,
};
pub use consumption::ConsumptionGuard;

pub fn cpu_cap_80(_: &crate::format::model_config::TrainingMetrics) -> f64 {
    ConsumptionGuard::detect().compute_cap()
}

fn dot_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
    let mut i = 0usize;
    while i + 8 <= n {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        acc4 += a[i + 4] * b[i + 4];
        acc5 += a[i + 5] * b[i + 5];
        acc6 += a[i + 6] * b[i + 6];
        acc7 += a[i + 7] * b[i + 7];
        i += 8;
    }
    let mut acc = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

fn axpy_scalar(dst: &mut [f32], a: f32, src: &[f32], n: usize) {
    let mut i = 0usize;
    while i + 8 <= n {
        dst[i] += a * src[i];
        dst[i + 1] += a * src[i + 1];
        dst[i + 2] += a * src[i + 2];
        dst[i + 3] += a * src[i + 3];
        dst[i + 4] += a * src[i + 4];
        dst[i + 5] += a * src[i + 5];
        dst[i + 6] += a * src[i + 6];
        dst[i + 7] += a * src[i + 7];
        i += 8;
    }
    while i < n {
        dst[i] += a * src[i];
        i += 1;
    }
}

pub(crate) fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::simd::avx2_fma_available() {
            return unsafe { x86::simd::dot_avx(a, b, n) };
        }
    }
    dot_scalar(a, b, n)
}

pub(crate) fn dot_f64(a: &[f64], b: &[f64], n: usize) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::simd::avx2_fma_available() {
            return unsafe { x86::simd::dot_avx_f64(a, b, n) };
        }
    }
    let mut acc = 0.0f64;
    let mut i = 0usize;
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

pub(crate) fn axpy_f64(dst: &mut [f64], a: f64, src: &[f64], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::simd::avx2_fma_available() {
            unsafe { x86::simd::axpy_avx_f64(dst, a, src, n) };
            return;
        }
    }
    let mut i = 0usize;
    while i < n {
        dst[i] += a * src[i];
        i += 1;
    }
}pub(crate) fn l2_cache_bytes() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        return x86::simd::l2_cache_bytes();
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        0
    }
}

pub(crate) fn dot<T: crate::base::math::Float>(a: &[T], b: &[T], n: usize) -> T {
    if T::DTYPE_TAG == 0 {
        let af = unsafe { core::slice::from_raw_parts(a.as_ptr().cast::<f32>(), a.len()) };
        let bf = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<f32>(), b.len()) };
        T::from_f32(dot_f32(af, bf, n))
    } else if T::DTYPE_TAG == 1 {
        let ad = unsafe { core::slice::from_raw_parts(a.as_ptr().cast::<f64>(), a.len()) };
        let bd = unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<f64>(), b.len()) };
        T::from_f64(dot_f64(ad, bd, n))
    } else {
        let mut acc = T::ZERO;
        let mut i = 0;
        while i < n {
            acc += a[i] * b[i];
            i += 1;
        }
        acc
    }
}

pub(crate) fn axpy_f32(dst: &mut [f32], a: f32, src: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if x86::simd::avx2_fma_available() {
            unsafe { x86::simd::axpy_avx(dst, a, src, n) };
            return;
        }
    }
    axpy_scalar(dst, a, src, n)
}

pub(crate) fn axpy<T: crate::base::math::Float>(dst: &mut [T], a: T, src: &[T], n: usize) {
    if T::DTYPE_TAG == 0 {
        let df = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), dst.len()) };
        let sf = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<f32>(), src.len()) };
        axpy_f32(df, a.to_f64() as f32, sf, n);
    } else if T::DTYPE_TAG == 1 {
        let dd = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f64>(), dst.len()) };
        let sd = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<f64>(), src.len()) };
        axpy_f64(dd, a.to_f64(), sd, n);
    } else {
        let mut i = 0;
        while i < n {
            dst[i] += a * src[i];
            i += 1;
        }
    }
}
