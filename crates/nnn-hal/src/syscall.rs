use nnn_config::ModelConfig;

pub fn initialize<T>(config: &ModelConfig<T>) {
    crate::guardian::initialize(config);
}

pub const O_RDONLY: i32 = crate::guardian::O_RDONLY;
pub const O_WRONLY: i32 = crate::guardian::O_WRONLY;
pub const O_APPEND: i32 = crate::guardian::O_APPEND;

pub fn o_creat() -> i32 { crate::guardian::o_creat() }
pub fn o_trunc() -> i32 { crate::guardian::o_trunc() }

pub fn sys_open(path: &[u8], flags: i32, mode: u32) -> i64 {
    crate::guardian::open(path, flags, mode)
}

pub fn sys_close(fd: i64) -> i64 {
    crate::guardian::close(fd)
}

pub fn sys_write_fd(fd: i64, buf: &[u8]) -> i64 {
    crate::guardian::write(fd, buf)
}

pub fn sys_read_fd(fd: i64, buf: &mut [u8]) -> i64 {
    crate::guardian::read(fd, buf)
}

pub fn sys_mkdir(path: &[u8], mode: u32) -> i64 {
    crate::guardian::mkdir(path, mode)
}

pub fn mmap_shared_anon(size: usize) -> *mut u8 {
    crate::guardian::mmap(size)
}

pub fn mmap_private_anon(size: usize) -> *mut u8 {
    crate::guardian::mmap_private(size)
}

pub fn munmap(ptr: *mut u8, size: usize) {
    crate::guardian::munmap(ptr, size)
}

pub fn monotonic_ns() -> u64 {
    crate::guardian::monotonic_ns()
}

pub fn fork() -> i64 {
    crate::guardian::fork()
}

pub fn waitpid(pid: i64) {
    crate::guardian::waitpid(pid)
}

pub fn exit(code: i32) -> ! {
    crate::guardian::exit(code)
}

pub fn set_affinity(mask: usize) {
    crate::guardian::set_affinity(mask)
}

#[repr(C)]
pub struct DirEnt64 {
    pub ino: u64,
    pub off: i64,
    pub reclen: u16,
    pub ftype: u8,
    pub name: [u8; 256],
}

pub fn sys_getdents64(fd: i64, buf: &mut [u8]) -> i64 {
    crate::guardian::getdents(fd, buf)
}

pub fn gpu_device_probe() -> Option<(u32, u32, u32)> {
    crate::guardian::gpu_device_probe()
}

pub fn gpu_gem_selftest(elements: usize) -> Option<(u32, usize, bool, bool)> {
    crate::guardian::gpu_gem_selftest(elements)
}

pub fn gpu_compute_selftest(elements: usize) -> Option<(usize, f32, f32, bool)> {
    crate::guardian::gpu_compute_selftest(elements)
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul(
    src: &[u8], dst: &mut [u8], batch_size: usize, stride: usize, in_size: usize,
    out_size: usize, weights: &[u8], biases: &[u8], activation: u32,
) -> bool {
    gpu_matmul_with_precision(
        nnn_config::Precision::F32, src, dst, batch_size, stride, in_size, out_size, weights,
        biases, activation,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul_with_precision(
    precision: nnn_config::Precision,
    src: &[u8], dst: &mut [u8], batch_size: usize, stride: usize, in_size: usize,
    out_size: usize, weights: &[u8], biases: &[u8], activation: u32,
) -> bool {
    crate::guardian::gpu_matmul_with_precision(
        precision, src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
    )
}

pub fn gpu_gemm(
    elem_size: usize, x: &[u8], y: &mut [u8], rows: usize, in_d: usize, out_d: usize,
    weights: &[u8],
) -> bool {
    crate::guardian::gpu_gemm(elem_size, x, y, rows, in_d, out_d, weights)
}

pub fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    crate::guardian::dot_f32(a, b, n)
}

pub fn dot_f64(a: &[f64], b: &[f64], n: usize) -> f64 {
    crate::guardian::dot_f64(a, b, n)
}

pub fn l2_cache_bytes() -> usize {
    crate::guardian::l2_cache_bytes()
}

pub fn axpy_f32(dst: &mut [f32], factor: f32, src: &[f32], n: usize) {
    crate::guardian::axpy_f32(dst, factor, src, n)
}

pub fn axpy_f64(dst: &mut [f64], factor: f64, src: &[f64], n: usize) {
    crate::guardian::axpy_f64(dst, factor, src, n)
}

pub fn futex_wait_u32(addr: *const u32, value: u32) {
    crate::guardian::futex_wait(addr, value)
}

pub fn futex_wake_u32(addr: *const u32, count: u32) {
    crate::guardian::futex_wake(addr, count)
}

pub fn prctl_set_pdeathsig(sig: usize) {
    crate::guardian::prctl(sig)
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
pub unsafe fn clone_thread(
    flags: usize, stack: usize, entry: extern "C" fn(usize), arg: usize, done: usize,
) -> i64 {
    crate::guardian::clone_thread(flags, stack, entry, arg, done)
}

pub fn allow_gpu(current_pct: u64, requested_pct: u64) -> bool {
    crate::guardian::allow_accelerator(crate::guardian::Accelerator::Gpu, current_pct, requested_pct)
}

pub fn allow_tpu(current_pct: u64, requested_pct: u64) -> bool {
    crate::guardian::allow_accelerator(crate::guardian::Accelerator::Tpu, current_pct, requested_pct)
}

pub fn allow_lpu(current_pct: u64, requested_pct: u64) -> bool {
    crate::guardian::allow_accelerator(crate::guardian::Accelerator::Lpu, current_pct, requested_pct)
}

pub fn adjusted_samples_per_cycle(
    base_batch: usize,
    profile: crate::arch::HardwareProfile,
) -> usize {
    crate::guardian::adjusted_samples_per_cycle(base_batch, profile)
}

pub struct SwapBatch {
    ptr: *mut u8,
    bytes: usize,
    input_size: usize,
    output_size: usize,
    count: usize,
    max_count: usize,
}

unsafe impl Send for SwapBatch {}
unsafe impl Sync for SwapBatch {}

impl SwapBatch {
    pub fn new(input_size: usize, output_size: usize, max_count: usize) -> Option<Self> {
        let sample_bytes = (input_size.saturating_add(output_size)).saturating_mul(4);
        let bytes = sample_bytes.saturating_mul(max_count);
        if bytes == 0 { return None; }
        let ptr = mmap_shared_anon(bytes);
        if ptr.is_null() { return None; }
        Some(Self { ptr, bytes, input_size, output_size, count: 0, max_count })
    }

    pub fn input_size(&self) -> usize { self.input_size }
    pub fn output_size(&self) -> usize { self.output_size }
    pub fn sample_bytes(&self) -> usize { (self.input_size + self.output_size) * 4 }
    pub fn max_count(&self) -> usize { self.max_count }
    pub fn set_count(&mut self, count: usize) { self.count = count.min(self.max_count); }
    pub fn ptr(&mut self) -> *mut u8 { self.ptr }

    pub fn get_sample(&self, i: usize) -> (&[f32], &[f32]) {
        let offset = i * self.sample_bytes();
        unsafe {
            let input = core::slice::from_raw_parts(self.ptr.add(offset) as *const f32, self.input_size);
            let target = core::slice::from_raw_parts(self.ptr.add(offset + self.input_size * 4) as *const f32, self.output_size);
            (input, target)
        }
    }

    pub fn release(&mut self) {
        if !self.ptr.is_null() {
            munmap(self.ptr, self.bytes);
            self.ptr = core::ptr::null_mut();
        }
    }
}
