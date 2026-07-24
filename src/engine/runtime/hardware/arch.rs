use crate::engine::{
	active_backend_contract_capabilities,
	active_backend_detected_flop_abstraction,
	active_backend_detected_ram_abstraction,
	build_ram_abstraction,
	clear_backend_detected_profile,
	clear_hardware_snapshot,
	get_compute_backend,
	host_os_flags_from_name,
	probe_hardware_snapshot,
	set_hardware_headroom_ppm,
	set_hardware_snapshot,
	BackendContractCapabilities,
	ComputeBackend,
	HardwareComponentSnapshot,
	HardwareDetectionSnapshot,
	DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND,
	DEFAULT_GPU_TOTAL_MEMORY_BYTES,
	DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND,
	DEFAULT_LPU_TOTAL_MEMORY_BYTES,
	DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND,
	DEFAULT_TPU_TOTAL_MEMORY_BYTES,
	HARDWARE_TIMING_FLAG_CLOCK_HZ_VALID,
	HARDWARE_TIMING_FLAG_CYCLE_COUNTER_VALID,
	HARDWARE_TIMING_FLAG_NOMINAL_CYCLES_VALID,
	HARDWARE_TIMING_FLAG_OBSERVED_CYCLES_VALID,
	HOST_OS_FLAG_LINUX,
	HOST_OS_FLAG_MACOS,
	HOST_OS_FLAG_WINDOWS,
	KERNEL_SPINLOCK_FLAG,
	KERNEL_SPINLOCK_FLAG_ATTENTION,
	KERNEL_SPINLOCK_FLAG_GLOBAL,
	KERNEL_SPINLOCK_FLAG_LAYER_NORM,
	KERNEL_SPINLOCK_FLAG_OPTIMIZER,
	KERNEL_SPINLOCK_FLAG_OS_LINUX_HINT,
	KERNEL_SPINLOCK_FLAG_OS_MACOS_HINT,
	KERNEL_SPINLOCK_FLAG_OS_WINDOWS_HINT,
	KERNEL_SPINLOCK_FLAG_QUANTIZATION,
	KERNEL_SPINLOCK_FLAG_RMS_NORM,
	KERNEL_SPINLOCK_FLAG_SOFTMAX,
};
use crate::engine::runtime::RuntimeProfile;
use core::sync::atomic::{AtomicUsize, Ordering};
use super::types::{
	BackendCapabilities,
	HardwareProfile,
	ResourceSnapshot,
	TrainingRuntimePlan,
};

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
	crate::engine::runtime::parallel::initialize_parallel_runtime();
	set_cpu_usage_reader(default_cpu_usage_reader);
	let workers = detected_parallelism();
	let _process_name = process_identifier();
	let sig = os_probe(workers);
	HARDWARE_INIT_SIG.store(sig as usize, Ordering::Release);
	crate::engine::cpu_kernels::install();
	crate::engine::gpu_kernels::install();
	#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
	{
		crate::engine::tpu_kernels::install();
		crate::engine::lpu_kernels::install();
	}
	clear_backend_detected_profile(get_compute_backend());
	clear_hardware_snapshot();
	let consumption = super::consumption::ConsumptionGuard::from_profile(detect_hardware_profile());
	set_hardware_headroom_ppm(consumption.accelerator_headroom_ppm(get_compute_backend()));
	crate::engine::runtime::set_precision(crate::engine::runtime::Precision::F32);
	let _ = crate::engine::runtime::gpu_gem_selftest(256);
	let _ = crate::engine::runtime::gpu_compute_selftest(256);
	let mut profile_scratch = [0u8; 16];
	let _ = crate::engine::runtime::gpu_matmul_profile(crate::engine::runtime::GpuMatmulProfileArgs {
		src: &[],
		dst: &mut profile_scratch,
		batch_size: 0,
		stride: 0,
		in_size: 0,
		out_size: 0,
		weights: &[],
		biases: &[],
		activation: 0,
	});
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
	if get_compute_backend() != ComputeBackend::Cpu {
		if let Some(flop) = active_backend_detected_flop_abstraction() {
			let compute_scale = saturating_u128_to_usize(flop.tflops_x1000 / 1000).max(1);
			bounded = bounded.saturating_mul(compute_scale.min(8));
		}
		if let Some(accel_ram) = active_backend_detected_ram_abstraction() {
			let accel_cap = saturating_u128_to_usize(accel_ram.available_mib).max(1);
			bounded = bounded.min(accel_cap);
		}
	}
	bounded.max(1)
}

pub fn adjusted_samples_per_cycle(base_batch: usize, profile: HardwareProfile) -> usize {
	base_batch.max(recommended_train_samples_per_cycle(profile)).max(1)
}

pub fn derive_runtime_plan(profile: HardwareProfile) -> TrainingRuntimePlan {
	let guard = super::consumption::ConsumptionGuard::from_profile(profile);
	let max_workers = guard.clamp_workers(profile.cores.max(1));
	let batch_size = recommended_train_samples_per_cycle(profile).max(max_workers);
	TrainingRuntimePlan { batch_size, max_workers }
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

	ResourceSnapshot {
		cpu_usage: current_cpu_usage(),
		ram_total: profile.ram_total,
		ram_available: profile.ram_available,
		gpu_dispatches,
		tpu_dispatches,
		lpu_dispatches,
	}
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
	let backend = get_compute_backend();
	if backend == ComputeBackend::Cpu {
		return None;
	}
	active_backend_contract_capabilities().map(map_backend_capabilities)
}
