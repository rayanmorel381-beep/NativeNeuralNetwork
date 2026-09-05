use nnn_config::ModelConfig;
use core::sync::atomic::{AtomicU8, Ordering};

mod capacity;
mod gate;
mod reaper;
mod surge;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Accelerator {
    Gpu,
    Tpu,
    Lpu,
}

static CPU_LIMIT_PCT: AtomicU8 = AtomicU8::new(100);
static RAM_LIMIT_PCT: AtomicU8 = AtomicU8::new(100);

pub(crate) const O_RDONLY: i32 = crate::arch::SYSCALL_O_RDONLY;
pub(crate) const O_WRONLY: i32 = crate::arch::SYSCALL_O_WRONLY;
pub(crate) const O_APPEND: i32 = crate::arch::SYSCALL_O_APPEND;
pub(crate) fn o_creat() -> i32 { crate::arch::syscall_o_creat() }
pub(crate) fn o_trunc() -> i32 { crate::arch::syscall_o_trunc() }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GuardianPolicy {
    cpu_limit_pct: u8,
    gpu_limit_pct: u8,
    tpu_limit_pct: u8,
    lpu_limit_pct: u8,
    ram_limit_pct: u8,
}

pub(crate) fn policy_from_config<T>(config: &ModelConfig<T>) -> GuardianPolicy {
    let profile = config.runtime.profile;
    GuardianPolicy {
        cpu_limit_pct: profile.max_cpu_pct(),
        gpu_limit_pct: profile.max_gpu_pct(),
        tpu_limit_pct: profile.max_tpu_pct(),
        lpu_limit_pct: profile.max_lpu_pct(),
        ram_limit_pct: profile.max_ram_pct(),
    }
}

pub(crate) fn initialize<T>(config: &ModelConfig<T>) -> GuardianPolicy {
    super::arch::ensure_hardware_init();
    let policy = policy_from_config(config);
    CPU_LIMIT_PCT.store(policy.cpu_limit_pct(), Ordering::Release);
    RAM_LIMIT_PCT.store(policy.ram_limit_pct(), Ordering::Release);
    surge::set_cpu_budget(u64::from(policy.cpu_limit_pct()));
    surge::set_memory_budget(u64::from(policy.ram_limit_pct()));
    capacity::set_cpu_reader(|| 0);
    capacity::set_memory_reader(|| 0);
    capacity::set_swap_reader(|| 0);
    reaper::mark_activity(0);
    policy
}

pub(crate) fn authorize_syscall() -> bool {
    let snapshot = crate::arch::resource_snapshot();
    let cpu_pct = capacity::cpu_usage()
        .unwrap_or((snapshot.cpu_usage.max(0.0) * 100.0) as u64);
    let _swap_usage = capacity::swap_usage();
    let ram_pct: u64 = if snapshot.ram_total == 0 {
        0
    } else {
        snapshot.ram_total.saturating_sub(snapshot.ram_available)
            .saturating_mul(100) / snapshot.ram_total
    } as u64;
    let now = crate::arch::syscall_monotonic_ns();
    let _expired = reaper::expired(now);
    reaper::mark_activity(now);
    let policy = GuardianPolicy {
        cpu_limit_pct: CPU_LIMIT_PCT.load(Ordering::Acquire),
        gpu_limit_pct: 100,
        tpu_limit_pct: 100,
        lpu_limit_pct: 100,
        ram_limit_pct: RAM_LIMIT_PCT.load(Ordering::Acquire),
    };
    gate::cpu(policy, cpu_pct, 0) && gate::memory(policy, ram_pct, 0)
}

pub(crate) fn open(path: &[u8], flags: i32, mode: u32) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_open(path, flags, mode)
}
pub(crate) fn close(fd: i64) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_close(fd)
}
pub(crate) fn write(fd: i64, buf: &[u8]) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_write(fd, buf)
}
pub(crate) fn read(fd: i64, buf: &mut [u8]) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_read(fd, buf)
}
pub(crate) fn mkdir(path: &[u8], mode: u32) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_mkdir(path, mode)
}
pub(crate) fn mmap(size: usize) -> *mut u8 {
    if !authorize_syscall() { return core::ptr::null_mut(); }
    crate::arch::syscall_mmap(size)
}
pub(crate) fn mmap_private(size: usize) -> *mut u8 {
    if !authorize_syscall() { return core::ptr::null_mut(); }
    crate::arch::syscall_mmap_private(size)
}
pub(crate) fn munmap(ptr: *mut u8, size: usize) {
    if authorize_syscall() { crate::arch::syscall_munmap(ptr, size); }
}
pub(crate) fn monotonic_ns() -> u64 { crate::arch::syscall_monotonic_ns() }
pub(crate) fn fork() -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_fork()
}
pub(crate) fn waitpid(pid: i64) {
    if authorize_syscall() { crate::arch::syscall_waitpid(pid); }
}
pub(crate) fn exit(code: i32) -> ! {
    if !authorize_syscall() { loop { core::hint::spin_loop(); } }
    crate::arch::syscall_exit(code)
}
pub(crate) fn set_affinity(mask: usize) {
    if authorize_syscall() { crate::arch::syscall_set_affinity(mask); }
}
pub(crate) fn getdents(fd: i64, buf: &mut [u8]) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_getdents(fd, buf)
}

pub(crate) fn gpu_device_probe() -> Option<(u32, u32, u32)> {
    if !authorize_syscall() { return None; }
    crate::arch::gpu_device_probe()
}

pub(crate) fn gpu_gem_selftest(elements: usize) -> Option<(u32, usize, bool, bool)> {
    if !authorize_syscall() { return None; }
    crate::arch::gpu_gem_selftest(elements)
}

pub(crate) fn gpu_compute_selftest(elements: usize) -> Option<(usize, f32, f32, bool)> {
    if !authorize_syscall() { return None; }
    crate::arch::gpu_compute_selftest(elements)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_matmul_with_precision(
    precision: nnn_config::Precision,
    src: &[u8], dst: &mut [u8], batch_size: usize, stride: usize, in_size: usize,
    out_size: usize, weights: &[u8], biases: &[u8], activation: u32,
) -> bool {
    if !authorize_syscall() { return false; }
    crate::arch::gpu_matmul_with_precision(
        precision, src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
    )
}

pub(crate) fn gpu_gemm(
    elem_size: usize, x: &[u8], y: &mut [u8], rows: usize, in_d: usize, out_d: usize,
    weights: &[u8],
) -> bool {
    if !authorize_syscall() { return false; }
    crate::arch::gpu_gemm(elem_size, x, y, rows, in_d, out_d, weights)
}

pub(crate) fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    if !authorize_syscall() { return 0.0; }
    crate::arch::dot_f32(a, b, n)
}

pub(crate) fn dot_f64(a: &[f64], b: &[f64], n: usize) -> f64 {
    if !authorize_syscall() { return 0.0; }
    crate::arch::dot_f64(a, b, n)
}

pub(crate) fn l2_cache_bytes() -> usize {
    if !authorize_syscall() { return 0; }
    crate::arch::l2_cache_bytes()
}

pub(crate) fn axpy_f32(dst: &mut [f32], factor: f32, src: &[f32], n: usize) {
    if authorize_syscall() { crate::arch::axpy_f32(dst, factor, src, n); }
}

pub(crate) fn axpy_f64(dst: &mut [f64], factor: f64, src: &[f64], n: usize) {
    if authorize_syscall() { crate::arch::axpy_f64(dst, factor, src, n); }
}

pub(crate) fn futex_wait(addr: *const u32, value: u32) {
    if authorize_syscall() { crate::arch::syscall_futex_wait(addr, value); }
}

pub(crate) fn futex_wake(addr: *const u32, count: u32) {
    if authorize_syscall() { crate::arch::syscall_futex_wake(addr, count); }
}

pub(crate) fn prctl(sig: usize) {
    if authorize_syscall() { crate::arch::syscall_prctl(sig); }
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
pub(crate) unsafe fn clone_thread(
    flags: usize, stack: usize, entry: extern "C" fn(usize), arg: usize, done: usize,
) -> i64 {
    if !authorize_syscall() { return -1; }
    crate::arch::syscall_clone(flags, stack, entry, arg, done)
}

pub(crate) fn allow_accelerator(backend: Accelerator, current_pct: u64, requested_pct: u64) -> bool {
    let policy = GuardianPolicy {
        cpu_limit_pct: CPU_LIMIT_PCT.load(Ordering::Acquire),
        gpu_limit_pct: 100,
        tpu_limit_pct: 100,
        lpu_limit_pct: 100,
        ram_limit_pct: RAM_LIMIT_PCT.load(Ordering::Acquire),
    };
    gate::accelerator(policy, backend, current_pct, requested_pct)
}

pub(crate) fn adjusted_samples_per_cycle(
    base_batch: usize,
    profile: crate::arch::HardwareProfile,
) -> usize {
    crate::arch::adjusted_samples_per_cycle(base_batch, profile)
}

impl GuardianPolicy {
    pub(crate) fn cpu_limit_pct(self) -> u8 { self.cpu_limit_pct }
    pub(crate) fn gpu_limit_pct(self) -> u8 { self.gpu_limit_pct }
    pub(crate) fn tpu_limit_pct(self) -> u8 { self.tpu_limit_pct }
    pub(crate) fn lpu_limit_pct(self) -> u8 { self.lpu_limit_pct }
    pub(crate) fn ram_limit_pct(self) -> u8 { self.ram_limit_pct }
}
