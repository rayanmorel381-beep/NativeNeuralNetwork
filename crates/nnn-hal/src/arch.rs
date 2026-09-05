use core::sync::atomic::{AtomicUsize, Ordering};
pub use super::types::{
	BackendCapabilities,
	BackendContractCapabilities,
	HardwareComponentSnapshot,
	HardwareDetectionSnapshot,
	HardwareProfile,
	RamAbstraction,
	ResourceSnapshot,
	RuntimeProfile,
	TrainingRuntimePlan,
};

#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
use super::arm::linux::syscall as syscall_backend;
#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_os = "macos"))]
use super::arm::macos::syscall as syscall_backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
use super::x86::linux::syscall as syscall_backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "macos"))]
use super::x86::macos::syscall as syscall_backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "windows"))]
use super::x86::windows::syscall as syscall_backend;

pub(crate) fn syscall_open(path: &[u8], flags: i32, mode: u32) -> i64 {
	syscall_backend::sys_open(path, flags, mode)
}

pub(crate) const SYSCALL_O_RDONLY: i32 = syscall_backend::O_RDONLY;
pub(crate) const SYSCALL_O_WRONLY: i32 = syscall_backend::O_WRONLY;
pub(crate) const SYSCALL_O_APPEND: i32 = syscall_backend::O_APPEND;
pub(crate) fn syscall_o_creat() -> i32 { syscall_backend::O_CREAT }
pub(crate) fn syscall_o_trunc() -> i32 { syscall_backend::O_TRUNC }
pub(crate) fn syscall_close(fd: i64) -> i64 { syscall_backend::sys_close(fd) }
pub(crate) fn syscall_write(fd: i64, buf: &[u8]) -> i64 { syscall_backend::sys_write_fd(fd, buf) }
pub(crate) fn syscall_read(fd: i64, buf: &mut [u8]) -> i64 { syscall_backend::sys_read_fd(fd, buf) }
pub(crate) fn syscall_mkdir(path: &[u8], mode: u32) -> i64 { syscall_backend::sys_mkdir(path, mode) }
pub(crate) fn syscall_mmap(size: usize) -> *mut u8 { syscall_backend::mmap_shared_anon(size) }
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn syscall_mmap_private(size: usize) -> *mut u8 { syscall_backend::mmap_private_anon(size) }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn syscall_mmap_private(size: usize) -> *mut u8 { syscall_backend::mmap_shared_anon(size) }
pub(crate) fn syscall_munmap(ptr: *mut u8, size: usize) { syscall_backend::munmap(ptr, size) }
pub(crate) fn syscall_monotonic_ns() -> u64 { syscall_backend::monotonic_ns() }
pub(crate) fn syscall_fork() -> i64 { syscall_backend::fork() }
pub(crate) fn syscall_waitpid(pid: i64) { syscall_backend::waitpid(pid) }
pub(crate) fn syscall_exit(code: i32) -> ! { syscall_backend::exit(code) }
pub(crate) fn syscall_set_affinity(mask: usize) { syscall_backend::set_affinity(mask) }
pub(crate) fn syscall_getdents(fd: i64, buf: &mut [u8]) -> i64 {
	syscall_backend::sys_getdents64(fd, buf)
}
pub(crate) fn syscall_futex_wait(addr: *const u32, val: u32) { syscall_backend::futex_wait_u32(addr, val) }
pub(crate) fn syscall_futex_wake(addr: *const u32, n: u32) { syscall_backend::futex_wake_u32(addr, n) }
pub(crate) fn syscall_prctl(sig: usize) { syscall_backend::prctl_set_pdeathsig(sig) }

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
pub(crate) unsafe fn syscall_clone(
	flags: usize, stack: usize, entry: extern "C" fn(usize), arg: usize, done: usize,
) -> i64 {
	syscall_backend::clone_thread(flags, stack, entry, arg, done)
}

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_device_probe() -> Option<(u32, u32, u32)> { plat::linux::radeon_probe() }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_device_probe() -> Option<(u32, u32, u32)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_gem_selftest(elements: usize) -> Option<(u32, usize, bool, bool)> {
	plat::linux::radeon_gem_selftest(elements)
}
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_gem_selftest(_elements: usize) -> Option<(u32, usize, bool, bool)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_compute_selftest(elements: usize) -> Option<(usize, f32, f32, bool)> {
	plat::linux::radeon_gpu_compute_selftest(elements)
}
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_compute_selftest(_elements: usize) -> Option<(usize, f32, f32, bool)> { None }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_matmul_with_precision(
	precision: nnn_config::Precision,
	src: &[u8], dst: &mut [u8], batch_size: usize, stride: usize, in_size: usize,
	out_size: usize, weights: &[u8], biases: &[u8], activation: u32,
) -> bool {
	plat::linux::gpu_matmul_dispatch(
		precision, src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
	)
}
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_matmul_with_precision(
	_precision: nnn_config::Precision, _src: &[u8], _dst: &mut [u8], _batch_size: usize,
	_stride: usize, _in_size: usize, _out_size: usize, _weights: &[u8], _biases: &[u8],
	_activation: u32,
) -> bool { false }

#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
pub(crate) fn gpu_gemm(
	elem_size: usize, x: &[u8], y: &mut [u8], rows: usize, in_d: usize, out_d: usize,
	weights: &[u8],
) -> bool { plat::linux::gpu_gemm_dispatch(elem_size, x, y, rows, in_d, out_d, weights) }
#[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android"))))]
pub(crate) fn gpu_gemm(
	_elem_size: usize, _x: &[u8], _y: &mut [u8], _rows: usize, _in_d: usize, _out_d: usize,
	_weights: &[u8],
) -> bool { false }

#[cfg(target_arch = "x86_64")]
pub(crate) fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
	if super::x86::simd::avx2_fma_available() {
		return unsafe { super::x86::simd::dot_avx(a, b, n) };
	}
	let mut value = 0.0;
	for i in 0..n { value += a[i] * b[i]; }
	value
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
	let mut value = 0.0;
	for i in 0..n { value += a[i] * b[i]; }
	value
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn dot_f64(a: &[f64], b: &[f64], n: usize) -> f64 {
	if super::x86::simd::avx2_fma_available() {
		return unsafe { super::x86::simd::dot_avx_f64(a, b, n) };
	}
	let mut value = 0.0;
	for i in 0..n { value += a[i] * b[i]; }
	value
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn dot_f64(a: &[f64], b: &[f64], n: usize) -> f64 {
	let mut value = 0.0;
	for i in 0..n { value += a[i] * b[i]; }
	value
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn l2_cache_bytes() -> usize { super::x86::simd::l2_cache_bytes() }
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn l2_cache_bytes() -> usize { 0 }

const DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND: u64 = 1_000_000_000_000;
const DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND: u64 = 2_000_000_000_000;
const DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND: u64 = 250_000_000_000;
const DEFAULT_GPU_TOTAL_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DEFAULT_TPU_TOTAL_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const DEFAULT_LPU_TOTAL_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const HARDWARE_TIMING_FLAG_CLOCK_HZ_VALID: u64 = 1 << 0;
const HARDWARE_TIMING_FLAG_CYCLE_COUNTER_VALID: u64 = 1 << 1;
const HARDWARE_TIMING_FLAG_NOMINAL_CYCLES_VALID: u64 = 1 << 2;
const HARDWARE_TIMING_FLAG_OBSERVED_CYCLES_VALID: u64 = 1 << 3;
const HOST_OS_FLAG_LINUX: u32 = 1;
const HOST_OS_FLAG_WINDOWS: u32 = 2;
const HOST_OS_FLAG_MACOS: u32 = 3;
const KERNEL_SPINLOCK_FLAG: u64 = 1 << 0;
const KERNEL_SPINLOCK_FLAG_SOFTMAX: u64 = 1 << 1;
const KERNEL_SPINLOCK_FLAG_LAYER_NORM: u64 = 1 << 2;
const KERNEL_SPINLOCK_FLAG_RMS_NORM: u64 = 1 << 3;
const KERNEL_SPINLOCK_FLAG_ATTENTION: u64 = 1 << 4;
const KERNEL_SPINLOCK_FLAG_QUANTIZATION: u64 = 1 << 5;
const KERNEL_SPINLOCK_FLAG_OPTIMIZER: u64 = 1 << 6;
const KERNEL_SPINLOCK_FLAG_GLOBAL: u64 = 1 << 7;
const KERNEL_SPINLOCK_FLAG_OS_LINUX_HINT: u64 = 1 << 60;
const KERNEL_SPINLOCK_FLAG_OS_WINDOWS_HINT: u64 = 1 << 61;
const KERNEL_SPINLOCK_FLAG_OS_MACOS_HINT: u64 = 1 << 62;

fn build_ram_abstraction(total: u128, available: u128) -> RamAbstraction {
    let used = total.saturating_sub(available);
    RamAbstraction {
        total_bytes: total,
        available_bytes: available,
        used_bytes: used,
        total_mib: total / (1024 * 1024),
        available_mib: available / (1024 * 1024),
        used_mib: used / (1024 * 1024),
    }
}

fn host_os_flags_from_name(name: &str) -> u32 {
    match name {
        "linux" => HOST_OS_FLAG_LINUX,
        "windows" => HOST_OS_FLAG_WINDOWS,
        "macos" => HOST_OS_FLAG_MACOS,
        _ => 0,
    }
}

fn clear_backend_detected_profile() {}
fn clear_hardware_snapshot() {}
fn set_hardware_snapshot(_: HardwareDetectionSnapshot) {}
fn set_hardware_headroom_ppm(_: u32) {}
fn probe_hardware_snapshot() -> Option<HardwareDetectionSnapshot> { None }
fn active_backend_contract_capabilities() -> Option<BackendContractCapabilities> { None }

static HARDWARE_INIT_SIG: AtomicUsize = AtomicUsize::new(0);
static CPU_USAGE_READER: AtomicUsize = AtomicUsize::new(0);

#[cfg(any(target_os = "linux", target_os = "android"))]
const HOST_OS_NAME: &str = "linux";
#[cfg(target_os = "windows")]
const HOST_OS_NAME: &str = "windows";
#[cfg(target_os = "macos")]
const HOST_OS_NAME: &str = "macos";
#[cfg(not(any(
	target_os = "linux",
	target_os = "android",
	target_os = "windows",
	target_os = "macos"
)))]
const HOST_OS_NAME: &str = "unknown";

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::x86 as plat;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
use super::arm as plat;

pub fn detected_parallelism() -> usize {
	plat::detected_parallelism()
}

pub fn detected_frame_budget_us() -> u64 {
	plat::detected_frame_budget_us()
}

pub fn process_identifier() -> Option<&'static str> {
	plat::process_identifier()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_probe(workers: usize) -> bool {
	plat::linux_probe(workers)
}
#[cfg(target_os = "windows")]
fn os_probe(workers: usize) -> bool {
	plat::windows_probe(workers)
}
#[cfg(target_os = "macos")]
fn os_probe(workers: usize) -> bool {
	plat::macos_probe(workers)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_cpu_cores(c: usize) -> usize {
	plat::linux_cpu_cores(c)
}
#[cfg(target_os = "windows")]
fn os_cpu_cores(c: usize) -> usize {
	plat::windows_cpu_cores(c)
}
#[cfg(target_os = "macos")]
fn os_cpu_cores(c: usize) -> usize {
	plat::macos_cpu_cores(c)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_hardware_data(c: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
	plat::linux_hardware_data(c)
}
#[cfg(target_os = "windows")]
fn os_hardware_data(c: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
	plat::windows_hardware_data(c)
}
#[cfg(target_os = "macos")]
fn os_hardware_data(c: usize) -> (usize, usize, bool, bool, bool, usize, usize, usize) {
	plat::macos_hardware_data(c)
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn os_cpu_freq(cores: usize) -> (u32, u32) {
	plat::linux::live_cpu_freq_mhz(cores)
}
#[cfg(target_os = "windows")]
fn os_cpu_freq(cores: usize) -> (u32, u32) {
	plat::windows::windows_cpu_freq_mhz(cores)
}
#[cfg(target_os = "macos")]
fn os_cpu_freq(cores: usize) -> (u32, u32) {
	plat::macos::macos_cpu_freq_mhz(cores)
}

#[cfg(not(all(target_arch = "x86_64", target_os = "macos")))]
pub fn contains_ascii_nocase(haystack: &str, needle: &str) -> bool {
	let hay = haystack.as_bytes();
	let need = needle.as_bytes();
	if need.is_empty() {
		return true;
	}
	if hay.len() < need.len() {
		return false;
	}
	let end = hay.len() - need.len();
	let mut i = 0usize;
	while i <= end {
		let mut matched = true;
		let mut j = 0usize;
		while j < need.len() {
			if !hay[i + j].eq_ignore_ascii_case(&need[j]) {
				matched = false;
				break;
			}
			j += 1;
		}
		if matched {
			return true;
		}
		i += 1;
	}
	false
}

fn saturating_u128_to_usize(value: u128) -> usize {
	if value > usize::MAX as u128 {
		usize::MAX
	} else {
		value as usize
	}
}

fn host_kernel_spinlock_flags(os_flags: u32) -> u64 {
	let os_hint = match os_flags {
		HOST_OS_FLAG_LINUX => KERNEL_SPINLOCK_FLAG_OS_LINUX_HINT,
		HOST_OS_FLAG_WINDOWS => KERNEL_SPINLOCK_FLAG_OS_WINDOWS_HINT,
		HOST_OS_FLAG_MACOS => KERNEL_SPINLOCK_FLAG_OS_MACOS_HINT,
		_ => 0,
	};
	KERNEL_SPINLOCK_FLAG | KERNEL_SPINLOCK_FLAG_QUANTIZATION | os_hint
}

fn build_accelerator_component(
	total_memory_bytes: u64,
	sustained_flops_per_second: u64,
	kernel_spinlock_flags: u64,
	timing_flags: u64,
) -> HardwareComponentSnapshot {
	HardwareComponentSnapshot {
		present: 1,
		device_count: 1,
		total_memory_bytes,
		available_memory_bytes: total_memory_bytes.saturating_mul(85) / 100,
		sustained_flops_per_second,
		kernel_spinlock_flags,
		timing_flags,
		..HardwareComponentSnapshot::default()
	}
}

fn build_detected_hardware_snapshot(profile: HardwareProfile) -> HardwareDetectionSnapshot {
	let os_flags = host_os_flags_from_name(HOST_OS_NAME);
	let kernel_flags = host_kernel_spinlock_flags(os_flags)
		| KERNEL_SPINLOCK_FLAG_SOFTMAX
		| KERNEL_SPINLOCK_FLAG_LAYER_NORM
		| KERNEL_SPINLOCK_FLAG_RMS_NORM
		| KERNEL_SPINLOCK_FLAG_ATTENTION
		| KERNEL_SPINLOCK_FLAG_OPTIMIZER
		| KERNEL_SPINLOCK_FLAG_GLOBAL;
	let timing_flags = HARDWARE_TIMING_FLAG_CLOCK_HZ_VALID
		| HARDWARE_TIMING_FLAG_CYCLE_COUNTER_VALID
		| HARDWARE_TIMING_FLAG_NOMINAL_CYCLES_VALID
		| HARDWARE_TIMING_FLAG_OBSERVED_CYCLES_VALID;
	let mut snapshot = HardwareDetectionSnapshot {
		cpu_logical_cores: profile.cores as u32,
		os_flags,
		system_total_memory_bytes: profile.ram_total as u64,
		system_available_memory_bytes: profile.ram_available as u64,
		cpu_kernel_spinlock_flags: kernel_flags,
		cpu_timing_flags: timing_flags,
		..HardwareDetectionSnapshot::default()
	};
	if profile.gpu {
		snapshot.gpu = build_accelerator_component(
			DEFAULT_GPU_TOTAL_MEMORY_BYTES,
			DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND,
			kernel_flags,
			timing_flags,
		);
	}
	if profile.tpu {
		snapshot.tpu = build_accelerator_component(
			DEFAULT_TPU_TOTAL_MEMORY_BYTES,
			DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND,
			kernel_flags,
			timing_flags,
		);
	}
	if profile.lpu {
		snapshot.lpu = build_accelerator_component(
			DEFAULT_LPU_TOTAL_MEMORY_BYTES,
			DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND,
			kernel_flags,
			timing_flags,
		);
	}
	snapshot
}

pub fn set_cpu_usage_reader(reader: fn() -> f32) {
	CPU_USAGE_READER.store(reader as usize, Ordering::Release);
}

fn current_cpu_usage() -> f32 {
	let raw = CPU_USAGE_READER.load(Ordering::Acquire);
	if raw == 0 {
		0.0
	} else {
		unsafe { core::mem::transmute::<usize, fn() -> f32>(raw)() }
	}
}

fn default_cpu_usage_reader() -> f32 {
	0.0
}

pub fn ensure_hardware_init() {
	set_cpu_usage_reader(default_cpu_usage_reader);
	let workers = detected_parallelism();
	let _process_name = process_identifier();
	let sig = os_probe(workers);
	HARDWARE_INIT_SIG.store(sig as usize, Ordering::Release);
	clear_backend_detected_profile();
	clear_hardware_snapshot();
	set_hardware_headroom_ppm(0);
	let snapshot = build_detected_hardware_snapshot(detect_hardware_profile());
	set_hardware_snapshot(snapshot);
	if let Some(caps) = active_backend_capabilities() {
		let _ = caps.opcode_bits();
		let _ = caps.flops();
		let _ = caps.memory_budget_bytes();
	}
	let runtime_snapshot = resource_snapshot();
	let _ = runtime_snapshot.cpu_load();
	let _ = runtime_snapshot.memory_budget();
	let _ = runtime_snapshot.accelerator_dispatches();
	derive_runtime_profile(detect_hardware_profile());
}

pub fn detect_hardware_profile() -> HardwareProfile {
	let cores = detected_parallelism().max(1);
	let cpu_cores = os_cpu_cores(cores).max(1);
	let (ram_l, avail_l, gpu_l, tpu_l, lpu_l, ..) = os_hardware_data(cores);
	let (avg_mhz, max_mhz) = os_cpu_freq(cpu_cores);
	HardwareProfile {
		cores: cpu_cores,
		ram_total: ram_l,
		ram_available: avail_l,
		gpu: gpu_l,
		tpu: tpu_l,
		lpu: lpu_l,
		cpu_avg_mhz: avg_mhz,
		cpu_max_mhz: max_mhz,
	}
}

pub fn recommended_train_samples_per_cycle(profile: HardwareProfile) -> usize {
	let accelerator_scale = 1usize + profile.gpu as usize + profile.tpu as usize + profile.lpu as usize;
	let base = profile.cores.max(1).saturating_mul(8).saturating_mul(accelerator_scale);
	let mut bounded = base;
	if let Some(snapshot) = probe_hardware_snapshot() {
		let host_ram = build_ram_abstraction(
			snapshot.system_total_memory_bytes as u128,
			snapshot.system_available_memory_bytes as u128,
		);
		let host_cap = saturating_u128_to_usize(host_ram.available_mib).max(1);
		bounded = bounded.min(host_cap);
	}
	bounded.max(1)
}

pub fn adjusted_samples_per_cycle(base_batch: usize, profile: HardwareProfile) -> usize {
	base_batch.max(recommended_train_samples_per_cycle(profile)).max(1)
}

pub(crate) fn axpy_f32(dst: &mut [f32], factor: f32, src: &[f32], n: usize) {
	#[cfg(target_arch = "x86_64")]
	if super::x86::simd::avx2_fma_available() {
		unsafe { super::x86::simd::axpy_avx(dst, factor, src, n) };
		return;
	}
	for i in 0..n { dst[i] += factor * src[i]; }
}

pub(crate) fn axpy_f64(dst: &mut [f64], factor: f64, src: &[f64], n: usize) {
	#[cfg(target_arch = "x86_64")]
	if super::x86::simd::avx2_fma_available() {
		unsafe { super::x86::simd::axpy_avx_f64(dst, factor, src, n) };
		return;
	}
	for i in 0..n { dst[i] += factor * src[i]; }
}

pub fn derive_runtime_plan(profile: HardwareProfile) -> TrainingRuntimePlan {
	let max_workers = profile.cores.max(1);
	let batch_size = recommended_train_samples_per_cycle(profile).max(max_workers);
	TrainingRuntimePlan { batch_size }
}

pub fn derive_runtime_profile(profile: HardwareProfile) -> RuntimeProfile {
	let plan = derive_runtime_plan(profile);
	RuntimeProfile::fp32(plan.batch_size, 1)
}

pub fn resource_snapshot() -> ResourceSnapshot {
	let profile = detect_hardware_profile();
	let cores = profile.cores.max(1);
	let (.., gpu_d_l, tpu_d_l, lpu_d_l) = os_hardware_data(cores);
	let gpu_dispatches = gpu_d_l;
	let tpu_dispatches = tpu_d_l;
	let lpu_dispatches = lpu_d_l;

	super::consumption::snapshot(
		current_cpu_usage(),
		profile.ram_total,
		profile.ram_available,
		gpu_dispatches,
		tpu_dispatches,
		lpu_dispatches,
	)
}


fn map_backend_capabilities(caps: BackendContractCapabilities) -> BackendCapabilities {
	BackendCapabilities {
		opcode_mask: caps.opcode_mask as u64,
		sustained_flops_per_second: caps.sustained_flops_per_second as f64,
		total_memory_bytes: caps.total_memory_bytes as usize,
		available_memory_bytes: caps.available_memory_bytes as usize,
	}
}

pub fn active_backend_capabilities() -> Option<BackendCapabilities> {
	active_backend_contract_capabilities().map(map_backend_capabilities)
}
