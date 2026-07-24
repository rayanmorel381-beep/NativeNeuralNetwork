use core::cell::UnsafeCell;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use super::types::*;

pub(super) static BACKEND: AtomicU8 = AtomicU8::new(ComputeBackend::Cpu as u8);
static HARDWARE_CONTRACT_STRICT: AtomicU8 = AtomicU8::new(1);
static CONTRACT_RUNTIME_FLAGS: AtomicUsize = AtomicUsize::new(0);
static CONTRACT_REJECT_HANDLER: AtomicUsize = AtomicUsize::new(0);
static CONTRACT_REJECT_USER_CTX: AtomicUsize = AtomicUsize::new(0);
static LAST_CONTRACT_REJECT_REASON: AtomicUsize = AtomicUsize::new(0);
static LAST_CONTRACT_REJECT_BACKEND: AtomicU8 = AtomicU8::new(ComputeBackend::Cpu as u8);
static LAST_CONTRACT_REJECT_OPCODE: AtomicUsize = AtomicUsize::new(0);
pub(super) static GPU_KERNEL_F32: AtomicUsize = AtomicUsize::new(0);
pub(super) static GPU_KERNEL_F64: AtomicUsize = AtomicUsize::new(0);
pub(super) static CPU_KERNEL_F32: AtomicUsize = AtomicUsize::new(0);
pub(super) static CPU_KERNEL_F64: AtomicUsize = AtomicUsize::new(0);
pub(super) static GPU_COMMAND_HANDLER: AtomicUsize = AtomicUsize::new(0);
static GPU_COMMAND_USER_CTX: AtomicUsize = AtomicUsize::new(0);
pub(super) static GPU_OPCODE_MASK: AtomicUsize = AtomicUsize::new(0);
pub(super) static TPU_COMMAND_HANDLER: AtomicUsize = AtomicUsize::new(0);
static TPU_COMMAND_USER_CTX: AtomicUsize = AtomicUsize::new(0);
pub(super) static TPU_OPCODE_MASK: AtomicUsize = AtomicUsize::new(0);
pub(super) static LPU_COMMAND_HANDLER: AtomicUsize = AtomicUsize::new(0);
static LPU_COMMAND_USER_CTX: AtomicUsize = AtomicUsize::new(0);
pub(super) static LPU_OPCODE_MASK: AtomicUsize = AtomicUsize::new(0);
static GPU_SUSTAINED_FLOPS_PER_SECOND: AtomicUsize = AtomicUsize::new(0);
static TPU_SUSTAINED_FLOPS_PER_SECOND: AtomicUsize = AtomicUsize::new(0);
static LPU_SUSTAINED_FLOPS_PER_SECOND: AtomicUsize = AtomicUsize::new(0);
static GPU_TOTAL_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static TPU_TOTAL_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static LPU_TOTAL_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_AVAILABLE_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static TPU_AVAILABLE_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static LPU_AVAILABLE_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static CONTRACT_MIN_SUSTAINED_FLOPS_PER_SECOND: AtomicUsize = AtomicUsize::new(0);
static CONTRACT_MIN_AVAILABLE_MEMORY_BYTES: AtomicUsize = AtomicUsize::new(0);
static HARDWARE_HEADROOM_PPM: AtomicUsize = AtomicUsize::new(DEFAULT_HARDWARE_HEADROOM_PPM);
static HARDWARE_PROBE_HANDLER: AtomicUsize = AtomicUsize::new(0);
static HARDWARE_PROBE_USER_CTX: AtomicUsize = AtomicUsize::new(0);
static SOFTBUFFER_RUNTIME_FLAGS: AtomicUsize = AtomicUsize::new(
    (SOFTBUFFER_RUNTIME_FLAG_CPU_STOPPED
        | SOFTBUFFER_RUNTIME_FLAG_GPU_STOPPED
        | SOFTBUFFER_RUNTIME_FLAG_TPU_STOPPED
        | SOFTBUFFER_RUNTIME_FLAG_LPU_STOPPED
        | SOFTBUFFER_RUNTIME_FLAG_SNAPSHOT_MISSING) as usize,
);

struct SnapshotStore {
    lock: AtomicU8,
    valid: AtomicU8,
    snapshot: UnsafeCell<HardwareDetectionSnapshot>,
}

unsafe impl Sync for SnapshotStore {}

static SNAPSHOT_STORE: SnapshotStore = SnapshotStore {
    lock: AtomicU8::new(0),
    valid: AtomicU8::new(0),
    snapshot: UnsafeCell::new(HardwareDetectionSnapshot {
        cpu_logical_cores: 0,
        os_flags: HOST_OS_FLAG_UNKNOWN,
        hardware_flags: HARDWARE_FLAG_CPU,
        system_total_memory_bytes: 0,
        system_available_memory_bytes: 0,
        cpu_kernel_spinlock_flags: 0,
        cpu_clock_hz: 0,
        cpu_cycle_counter_hz: 0,
        cpu_nominal_cycles_per_kernel: 0,
        cpu_observed_cycles_per_kernel: 0,
        cpu_timing_flags: 0,
        gpu: HardwareComponentSnapshot {
            present: 0, reserved0: 0, reserved1: 0, reserved2: 0,
            device_count: 0, total_memory_bytes: 0, available_memory_bytes: 0,
            sustained_flops_per_second: 0, kernel_spinlock_flags: 0,
            clock_hz: 0, cycle_counter_hz: 0, nominal_cycles_per_kernel: 0,
            observed_cycles_per_kernel: 0, timing_flags: 0,
        },
        tpu: HardwareComponentSnapshot {
            present: 0, reserved0: 0, reserved1: 0, reserved2: 0,
            device_count: 0, total_memory_bytes: 0, available_memory_bytes: 0,
            sustained_flops_per_second: 0, kernel_spinlock_flags: 0,
            clock_hz: 0, cycle_counter_hz: 0, nominal_cycles_per_kernel: 0,
            observed_cycles_per_kernel: 0, timing_flags: 0,
        },
        lpu: HardwareComponentSnapshot {
            present: 0, reserved0: 0, reserved1: 0, reserved2: 0,
            device_count: 0, total_memory_bytes: 0, available_memory_bytes: 0,
            sustained_flops_per_second: 0, kernel_spinlock_flags: 0,
            clock_hz: 0, cycle_counter_hz: 0, nominal_cycles_per_kernel: 0,
            observed_cycles_per_kernel: 0, timing_flags: 0,
        },
    }),
};

#[inline]
fn acquire_snapshot_lock() {
    loop {
        if SNAPSHOT_STORE
            .lock
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            break;
        }
        spin_loop();
    }
}

#[inline]
fn release_snapshot_lock() {
    SNAPSHOT_STORE.lock.store(0, Ordering::Release);
}

fn load_snapshot() -> Option<HardwareDetectionSnapshot> {
    if SNAPSHOT_STORE.valid.load(Ordering::SeqCst) == 0 {
        return None;
    }
    acquire_snapshot_lock();
    let snapshot = unsafe { *SNAPSHOT_STORE.snapshot.get() };
    release_snapshot_lock();
    Some(snapshot)
}

fn store_snapshot(snapshot: HardwareDetectionSnapshot) {
    acquire_snapshot_lock();
    unsafe {
        *SNAPSHOT_STORE.snapshot.get() = snapshot;
    }
    SNAPSHOT_STORE.valid.store(1, Ordering::SeqCst);
    release_snapshot_lock();
}

#[inline]
fn decode_contract_reject_handler(ptr: usize) -> Option<ContractRejectHandler> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, ContractRejectHandler>(ptr) })
}

#[inline]
pub(super) fn decode_gpu_command_handler(ptr: usize) -> Option<GpuCommandHandler> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, GpuCommandHandler>(ptr) })
}

#[inline]
pub fn decode_gpu_kernel_f32(ptr: usize) -> Option<GpuKernelF32> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, GpuKernelF32>(ptr) })
}

#[inline]
pub fn decode_gpu_kernel_f64(ptr: usize) -> Option<GpuKernelF64> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, GpuKernelF64>(ptr) })
}

#[inline]
pub(super) fn decode_cpu_kernel_f32(ptr: usize) -> Option<GpuKernelF32> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, GpuKernelF32>(ptr) })
}

#[inline]
pub(super) fn decode_cpu_kernel_f64(ptr: usize) -> Option<GpuKernelF64> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, GpuKernelF64>(ptr) })
}

#[inline]
fn decode_hardware_probe_handler(ptr: usize) -> Option<HardwareProbeFn> {
    if ptr == 0 { return None; }
    Some(unsafe { core::mem::transmute::<usize, HardwareProbeFn>(ptr) })
}

#[inline]
fn sanitize_component(mut component: HardwareComponentSnapshot) -> HardwareComponentSnapshot {
    if component.present == 0 || component.device_count == 0 {
        component.present = 0;
        component.device_count = 0;
        component.total_memory_bytes = 0;
        component.available_memory_bytes = 0;
        component.sustained_flops_per_second = 0;
    } else if component.available_memory_bytes > component.total_memory_bytes {
        component.available_memory_bytes = component.total_memory_bytes;
    }
    component
}

fn normalize_hardware_snapshot(
    mut snapshot: HardwareDetectionSnapshot,
) -> Option<HardwareDetectionSnapshot> {
    if !is_valid_os_flag_value(snapshot.os_flags) {
        return None;
    }
    if snapshot.system_available_memory_bytes > snapshot.system_total_memory_bytes {
        snapshot.system_available_memory_bytes = snapshot.system_total_memory_bytes;
    }
    snapshot.gpu = sanitize_component(snapshot.gpu);
    snapshot.tpu = sanitize_component(snapshot.tpu);
    snapshot.lpu = sanitize_component(snapshot.lpu);
    let mut normalized_flags = HARDWARE_FLAG_CPU;
    if component_compute_ready(snapshot.gpu) { normalized_flags |= HARDWARE_FLAG_GPU; }
    if component_compute_ready(snapshot.tpu) { normalized_flags |= HARDWARE_FLAG_TPU; }
    if component_compute_ready(snapshot.lpu) { normalized_flags |= HARDWARE_FLAG_LPU; }
    snapshot.hardware_flags = normalized_flags;
    Some(snapshot)
}

#[inline]
fn cpu_buffers_ready(snapshot: &HardwareDetectionSnapshot) -> bool {
    snapshot.system_total_memory_bytes > 0
        && snapshot.system_available_memory_bytes > 0
        && snapshot.system_available_memory_bytes <= snapshot.system_total_memory_bytes
}

#[inline]
fn cpu_softbuffer_ready(snapshot: &HardwareDetectionSnapshot) -> bool {
    cpu_buffers_ready(snapshot) && snapshot.cpu_kernel_spinlock_flags != 0
}

#[inline]
fn component_softbuffer_ready(component: HardwareComponentSnapshot) -> bool {
    component_buffers_ready(component) && component.kernel_spinlock_flags != 0
}

fn compute_softbuffer_runtime_flags(snapshot: &HardwareDetectionSnapshot) -> u32 {
    let mut flags = 0u32;
    if !cpu_softbuffer_ready(snapshot) { flags |= SOFTBUFFER_RUNTIME_FLAG_CPU_STOPPED; }
    if !component_softbuffer_ready(snapshot.gpu) { flags |= SOFTBUFFER_RUNTIME_FLAG_GPU_STOPPED; }
    if !component_softbuffer_ready(snapshot.tpu) { flags |= SOFTBUFFER_RUNTIME_FLAG_TPU_STOPPED; }
    if !component_softbuffer_ready(snapshot.lpu) { flags |= SOFTBUFFER_RUNTIME_FLAG_LPU_STOPPED; }
    flags
}

fn set_softbuffer_runtime_flags(flags: u32) {
    SOFTBUFFER_RUNTIME_FLAGS.store(flags as usize, Ordering::SeqCst);
}

fn build_fallback_snapshot() -> HardwareDetectionSnapshot {
    let os_flags = match option_env!("CARGO_CFG_TARGET_OS") {
        Some("linux") => HOST_OS_FLAG_LINUX,
        Some("windows") => HOST_OS_FLAG_WINDOWS,
        Some("macos") => HOST_OS_FLAG_MACOS,
        _ => HOST_OS_FLAG_UNKNOWN,
    };
    let cores = match option_env!("CARGO_CFG_TARGET_ARCH") {
        Some("aarch64") => 8u32,
        Some("x86_64") => 8u32,
        Some("x86") => 4u32,
        _ => 4u32,
    };
    let kernel_flags = match os_flags {
        HOST_OS_FLAG_LINUX => KERNEL_SPINLOCK_FLAG | KERNEL_SPINLOCK_FLAG_OS_LINUX_HINT,
        HOST_OS_FLAG_WINDOWS => KERNEL_SPINLOCK_FLAG | KERNEL_SPINLOCK_FLAG_OS_WINDOWS_HINT,
        HOST_OS_FLAG_MACOS => KERNEL_SPINLOCK_FLAG | KERNEL_SPINLOCK_FLAG_OS_MACOS_HINT,
        _ => KERNEL_SPINLOCK_FLAG,
    };
    let mut gpu = HardwareComponentSnapshot::default();
    let mut tpu = HardwareComponentSnapshot::default();
    let mut lpu = HardwareComponentSnapshot::default();
    let arch = option_env!("CARGO_CFG_TARGET_ARCH").unwrap_or("unknown");

    if arch == "x86_64" || arch == "aarch64" {
        gpu.present = 1;
        gpu.device_count = 1;
        gpu.total_memory_bytes = DEFAULT_GPU_TOTAL_MEMORY_BYTES;
        gpu.available_memory_bytes = DEFAULT_GPU_TOTAL_MEMORY_BYTES.saturating_mul(85) / 100;
        gpu.sustained_flops_per_second = DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND;
        gpu.kernel_spinlock_flags = kernel_flags;
    }
    if arch == "aarch64" {
        tpu.present = 1;
        tpu.device_count = 1;
        tpu.total_memory_bytes = DEFAULT_TPU_TOTAL_MEMORY_BYTES;
        tpu.available_memory_bytes = DEFAULT_TPU_TOTAL_MEMORY_BYTES.saturating_mul(85) / 100;
        tpu.sustained_flops_per_second = DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND;
        tpu.kernel_spinlock_flags = kernel_flags;

        lpu.present = 1;
        lpu.device_count = 1;
        lpu.total_memory_bytes = DEFAULT_LPU_TOTAL_MEMORY_BYTES;
        lpu.available_memory_bytes = DEFAULT_LPU_TOTAL_MEMORY_BYTES.saturating_mul(85) / 100;
        lpu.sustained_flops_per_second = DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND;
        lpu.kernel_spinlock_flags = kernel_flags;
    }

    HardwareDetectionSnapshot {
        cpu_logical_cores: cores,
        os_flags,
        hardware_flags: HARDWARE_FLAG_CPU,
        system_total_memory_bytes: 8 * 1024 * 1024 * 1024,
        system_available_memory_bytes: 6 * 1024 * 1024 * 1024,
        cpu_kernel_spinlock_flags: kernel_flags,
        cpu_clock_hz: 2200000000,
        cpu_cycle_counter_hz: 2200000000,
        cpu_nominal_cycles_per_kernel: 0,
        cpu_observed_cycles_per_kernel: 0,
        cpu_timing_flags: HARDWARE_TIMING_FLAG_CLOCK_HZ_VALID | HARDWARE_TIMING_FLAG_CYCLE_COUNTER_VALID,
        gpu,
        tpu,
        lpu,
    }
}

fn default_hardware_probe_handler(out_snapshot: *mut HardwareDetectionSnapshot, _user_ctx: usize) -> u32 {
    if out_snapshot.is_null() {
        return 0;
    }
    let snapshot = load_snapshot().unwrap_or_else(build_fallback_snapshot);
    unsafe {
        *out_snapshot = snapshot;
    }
    1
}

fn register_default_hardware_probe_if_missing() {
    if HARDWARE_PROBE_HANDLER.load(Ordering::SeqCst) == 0 {
        register_hardware_probe(default_hardware_probe_handler, 0);
    }
}

fn apply_backend_runtime_probe(backend: ComputeBackend) {
    let os_flags = probe_hardware_os_flags();
    let probe_flags = probe_hardware_probe_flags();
    let softbuffer_flags = probe_softbuffer_runtime_flags();
    let available_flags = probe_hardware_available_flags();
    let cpu_spinlock_flags = probe_cpu_kernel_spinlock_flags();
    let hardware_spinlock_flags = probe_hardware_kernel_spinlock_flags(backend);
    if os_flags != 0 || probe_flags != 0 || softbuffer_flags != 0 {
        set_contract_runtime_flags(CONTRACT_FLAG_REQUIRE_FINITE_INPUTS | CONTRACT_FLAG_DETERMINISTIC_MATH);
    }
    if available_flags != 0 || cpu_spinlock_flags != 0 || hardware_spinlock_flags != 0 {
        set_contract_runtime_flags(CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT);
    }
    if let Some(mut timing) = probe_cpu_timing_abstraction().or_else(|| probe_hardware_timing_abstraction(backend)) {
        if timing.timing_flags != 0 {
            timing.timing_flags |= HARDWARE_TIMING_FLAG_CLOCK_HZ_VALID;
        }
        if timing.timing_flags != 0 {
            set_contract_runtime_flags(CONTRACT_FLAG_DETERMINISTIC_MATH);
        }
    }
    if let Some(staging) = probe_hardware_buffer_staging_profile(backend) {
        set_contract_min_sustained_flops_per_second(staging.progressive_validated_bytes as u128 * 16);
        set_contract_min_available_memory_bytes(staging.final_buffer_bytes as u128);
    } else {
        set_contract_min_sustained_flops_per_second(0);
        set_contract_min_available_memory_bytes(0);
    }
    let requested_bytes = request_buffer_with_hardware_approval(backend, 1 << 20, 4);
    let backend_present = backend_present_from_detected_hardware(backend);
    let reject_reason = last_contract_reject_reason();
    let reject_backend = last_contract_reject_backend();
    let reject_opcode = last_contract_reject_opcode();
    if requested_bytes.is_some() && backend_present {
        set_contract_runtime_flags(CONTRACT_FLAG_REQUIRE_FINITE_INPUTS | CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT);
    } else if reject_reason.is_some() || reject_backend.is_some() || reject_opcode.is_some() {
        set_contract_runtime_flags(CONTRACT_FLAG_DETERMINISTIC_MATH);
    }
}

fn run_hardware_probe() -> Option<HardwareDetectionSnapshot> {
    if let Some(snapshot) = load_snapshot() {
        set_softbuffer_runtime_flags(compute_softbuffer_runtime_flags(&snapshot));
        return Some(snapshot);
    }

    let handler_ptr = HARDWARE_PROBE_HANDLER.load(Ordering::SeqCst);
    if let Some(handler) = decode_hardware_probe_handler(handler_ptr) {
        let mut out = HardwareDetectionSnapshot::default();
        let user_ctx = HARDWARE_PROBE_USER_CTX.load(Ordering::SeqCst);
        let handler_status = handler(&mut out as *mut HardwareDetectionSnapshot, user_ctx);
    if handler_status == 0 {
        return None;
    }
        if let Some(normalized) = normalize_hardware_snapshot(out) {
            store_snapshot(normalized);
            set_softbuffer_runtime_flags(compute_softbuffer_runtime_flags(&normalized));
            return Some(normalized);
        }
    }

    let fallback = build_fallback_snapshot();
    if let Some(normalized) = normalize_hardware_snapshot(fallback) {
        store_snapshot(normalized);
        set_softbuffer_runtime_flags(compute_softbuffer_runtime_flags(&normalized));
        return Some(normalized);
    }
    None
}

#[inline]
fn is_valid_os_flag_value(os_flags: u32) -> bool {
    os_flags <= HOST_OS_FLAG_MACOS
}

#[inline]
fn snapshot_component(
    snapshot: &HardwareDetectionSnapshot,
    backend: ComputeBackend,
) -> Option<HardwareComponentSnapshot> {
    match backend {
        ComputeBackend::Cpu => None,
        ComputeBackend::Gpu => Some(snapshot.gpu),
        ComputeBackend::Tpu => Some(snapshot.tpu),
        ComputeBackend::Lpu => Some(snapshot.lpu),
    }
}

pub fn probe_hardware_os_flags() -> u32 {
    run_hardware_probe().map(|s| s.os_flags).unwrap_or(0)
}

pub fn probe_hardware_probe_flags() -> u32 {
    let mut flags = 0u32;
    if HARDWARE_PROBE_HANDLER.load(Ordering::SeqCst) != 0 {
        flags |= HARDWARE_PROBE_FLAG_HANDLER_REGISTERED;
    }
    if SNAPSHOT_STORE.valid.load(Ordering::SeqCst) != 0 {
        flags |= HARDWARE_PROBE_FLAG_SNAPSHOT_AVAILABLE;
    }
    if run_hardware_probe().is_some() {
        flags |= HARDWARE_PROBE_FLAG_SNAPSHOT_AVAILABLE;
    }
    flags
}

pub fn set_hardware_snapshot(snapshot: HardwareDetectionSnapshot) -> bool {
    let Some(normalized) = normalize_hardware_snapshot(snapshot) else {
        return false;
    };
    store_snapshot(normalized);
    sync_detected_hardware_profile_to_contract();
    sync_backend_detected_profile(get_compute_backend());
    true
}

pub fn clear_hardware_snapshot() {
    SNAPSHOT_STORE.valid.store(0, Ordering::SeqCst);
    clear_contract_reject_handler();
    clear_hardware_probe();
    clear_cpu_kernel_f32();
    clear_cpu_kernel_f64();
    set_softbuffer_runtime_flags(
        SOFTBUFFER_RUNTIME_FLAG_CPU_STOPPED
            | SOFTBUFFER_RUNTIME_FLAG_GPU_STOPPED
            | SOFTBUFFER_RUNTIME_FLAG_TPU_STOPPED
            | SOFTBUFFER_RUNTIME_FLAG_LPU_STOPPED
            | SOFTBUFFER_RUNTIME_FLAG_SNAPSHOT_MISSING,
    );
}

pub fn probe_softbuffer_runtime_flags() -> u32 {
    run_hardware_probe();
    SOFTBUFFER_RUNTIME_FLAGS.load(Ordering::SeqCst) as u32
}

pub fn probe_hardware_available_flags() -> u32 {
    run_hardware_probe().map(|s| s.hardware_flags).unwrap_or(0)
}

pub fn probe_cpu_kernel_spinlock_flags() -> u64 {
    run_hardware_probe()
        .map(|s| s.cpu_kernel_spinlock_flags)
        .unwrap_or(0)
}

pub fn probe_hardware_kernel_spinlock_flags(backend: ComputeBackend) -> u64 {
    let Some(snapshot) = run_hardware_probe() else { return 0; };
    match backend {
        ComputeBackend::Cpu => snapshot.cpu_kernel_spinlock_flags,
        ComputeBackend::Gpu => snapshot.gpu.kernel_spinlock_flags,
        ComputeBackend::Tpu => snapshot.tpu.kernel_spinlock_flags,
        ComputeBackend::Lpu => snapshot.lpu.kernel_spinlock_flags,
    }
}

pub fn probe_cpu_timing_abstraction() -> Option<BackendTimingAbstraction> {
    let snapshot = run_hardware_probe()?;
    Some(BackendTimingAbstraction {
        clock_hz: snapshot.cpu_clock_hz,
        cycle_counter_hz: snapshot.cpu_cycle_counter_hz,
        nominal_cycles_per_kernel: snapshot.cpu_nominal_cycles_per_kernel,
        observed_cycles_per_kernel: snapshot.cpu_observed_cycles_per_kernel,
        timing_flags: snapshot.cpu_timing_flags,
    })
}

pub fn probe_hardware_timing_abstraction(
    backend: ComputeBackend,
) -> Option<BackendTimingAbstraction> {
    let snapshot = run_hardware_probe()?;
    match backend {
        ComputeBackend::Cpu => Some(BackendTimingAbstraction {
            clock_hz: snapshot.cpu_clock_hz,
            cycle_counter_hz: snapshot.cpu_cycle_counter_hz,
            nominal_cycles_per_kernel: snapshot.cpu_nominal_cycles_per_kernel,
            observed_cycles_per_kernel: snapshot.cpu_observed_cycles_per_kernel,
            timing_flags: snapshot.cpu_timing_flags,
        }),
        ComputeBackend::Gpu => Some(BackendTimingAbstraction {
            clock_hz: snapshot.gpu.clock_hz,
            cycle_counter_hz: snapshot.gpu.cycle_counter_hz,
            nominal_cycles_per_kernel: snapshot.gpu.nominal_cycles_per_kernel,
            observed_cycles_per_kernel: snapshot.gpu.observed_cycles_per_kernel,
            timing_flags: snapshot.gpu.timing_flags,
        }),
        ComputeBackend::Tpu => Some(BackendTimingAbstraction {
            clock_hz: snapshot.tpu.clock_hz,
            cycle_counter_hz: snapshot.tpu.cycle_counter_hz,
            nominal_cycles_per_kernel: snapshot.tpu.nominal_cycles_per_kernel,
            observed_cycles_per_kernel: snapshot.tpu.observed_cycles_per_kernel,
            timing_flags: snapshot.tpu.timing_flags,
        }),
        ComputeBackend::Lpu => Some(BackendTimingAbstraction {
            clock_hz: snapshot.lpu.clock_hz,
            cycle_counter_hz: snapshot.lpu.cycle_counter_hz,
            nominal_cycles_per_kernel: snapshot.lpu.nominal_cycles_per_kernel,
            observed_cycles_per_kernel: snapshot.lpu.observed_cycles_per_kernel,
            timing_flags: snapshot.lpu.timing_flags,
        }),
    }
}

#[inline]
fn saturating_u128_to_usize(value: u128) -> usize {
    if value > usize::MAX as u128 { usize::MAX } else { value as usize }
}

#[inline]
fn apply_headroom(value: usize, headroom_ppm: usize) -> usize {
    let ppm = headroom_ppm.min(900_000);
    let kept_ppm = 1_000_000usize.saturating_sub(ppm) as u128;
    ((value as u128).saturating_mul(kept_ppm) / 1_000_000u128) as usize
}

pub fn set_hardware_headroom_ppm(headroom_ppm: u32) {
    HARDWARE_HEADROOM_PPM.store((headroom_ppm as usize).min(900_000), Ordering::SeqCst);
}

pub fn get_hardware_headroom_ppm() -> u32 {
    HARDWARE_HEADROOM_PPM.load(Ordering::SeqCst) as u32
}

pub fn register_hardware_probe(handler: HardwareProbeFn, user_ctx: usize) {
    HARDWARE_PROBE_USER_CTX.store(user_ctx, Ordering::SeqCst);
    HARDWARE_PROBE_HANDLER.store(handler as usize, Ordering::SeqCst);
    sync_detected_hardware_profile_to_contract();
    sync_backend_detected_profile(get_compute_backend());
}

pub fn clear_hardware_probe() {
    HARDWARE_PROBE_HANDLER.store(0, Ordering::SeqCst);
    HARDWARE_PROBE_USER_CTX.store(0, Ordering::SeqCst);
}

pub fn probe_hardware_snapshot() -> Option<HardwareDetectionSnapshot> {
    run_hardware_probe()
}

fn backend_present_from_detected_hardware(backend: ComputeBackend) -> bool {
    if backend == ComputeBackend::Cpu { return true; }
    let Some(snapshot) = run_hardware_probe() else { return false; };
    let Some(component) = snapshot_component(&snapshot, backend) else { return false; };
    component_softbuffer_ready(component)
}

#[inline]
fn component_buffers_ready(component: HardwareComponentSnapshot) -> bool {
    component.present != 0
        && component.device_count > 0
        && component.total_memory_bytes > 0
        && component.available_memory_bytes > 0
        && component.available_memory_bytes <= component.total_memory_bytes
}

#[inline]
fn component_compute_ready(component: HardwareComponentSnapshot) -> bool {
    component.present != 0 && component.device_count > 0
}

pub(super) fn backend_compute_present_from_detected_hardware(backend: ComputeBackend) -> bool {
    if backend == ComputeBackend::Cpu { return false; }
    let Some(snapshot) = run_hardware_probe() else { return false; };
    let Some(component) = snapshot_component(&snapshot, backend) else { return false; };
    component_compute_ready(component)
}

#[inline]
fn progressive_validate_staged_size(current_bytes: u64) -> u64 {
    if current_bytes <= 1 { current_bytes } else { current_bytes / 2 }
}

fn build_buffer_staging_profile(
    component: HardwareComponentSnapshot,
    soft_overhead_ppm: u32,
) -> Option<BufferStagingProfile> {
    if !component_buffers_ready(component) { return None; }
    if component.kernel_spinlock_flags == 0 {
        return Some(BufferStagingProfile {
            discovered_total_bytes: component.total_memory_bytes,
            discovered_available_bytes: component.available_memory_bytes,
            staged_candidate_bytes: component.available_memory_bytes,
            progressive_validated_bytes: component.available_memory_bytes,
            soft_overhead_bytes: component.available_memory_bytes,
            final_buffer_bytes: 0,
            sequence_count: 0,
            stage_flags: DISCOVERY_STAGE_STAGING_ACTIVE
                | DISCOVERY_STAGE_SOFTBUFFER_STOPPED
                | DISCOVERY_STAGE_FINALIZED,
        });
    }
    let discovered_total = component.total_memory_bytes;
    let discovered_available = component.available_memory_bytes.min(discovered_total);
    let staged_candidate = discovered_available;
    let mut stage_flags = DISCOVERY_STAGE_STAGING_ACTIVE;
    if component.kernel_spinlock_flags != 0 { stage_flags |= DISCOVERY_STAGE_SPINLOCK_GUARDED; }
    let mut progressive_validated = staged_candidate;
    let mut sequence_count = 0u32;
    while sequence_count < 4 && progressive_validated > 1 {
        progressive_validated = progressive_validate_staged_size(progressive_validated);
        sequence_count = sequence_count.saturating_add(1);
    }
    if sequence_count > 0 { stage_flags |= DISCOVERY_STAGE_PROGRESSIVE_VALIDATED; }
    let ppm = soft_overhead_ppm.min(900_000) as u64;
    let soft_overhead_bytes = progressive_validated.saturating_mul(ppm).saturating_div(1_000_000);
    let final_buffer_bytes = progressive_validated.saturating_sub(soft_overhead_bytes);
    stage_flags |= DISCOVERY_STAGE_SOFT_OVERHEAD_APPLIED;
    stage_flags |= DISCOVERY_STAGE_FINALIZED;
    Some(BufferStagingProfile {
        discovered_total_bytes: discovered_total,
        discovered_available_bytes: discovered_available,
        staged_candidate_bytes: staged_candidate,
        progressive_validated_bytes: progressive_validated,
        soft_overhead_bytes,
        final_buffer_bytes,
        sequence_count,
        stage_flags,
    })
}

fn build_cpu_buffer_staging_profile(
    snapshot: &HardwareDetectionSnapshot,
    soft_overhead_ppm: u32,
) -> Option<BufferStagingProfile> {
    let discovered_total = snapshot.system_total_memory_bytes;
    let discovered_available = snapshot.system_available_memory_bytes.min(discovered_total);
    if discovered_total == 0 || discovered_available == 0 { return None; }
    if snapshot.cpu_kernel_spinlock_flags == 0 {
        return Some(BufferStagingProfile {
            discovered_total_bytes: discovered_total,
            discovered_available_bytes: discovered_available,
            staged_candidate_bytes: discovered_available,
            progressive_validated_bytes: discovered_available,
            soft_overhead_bytes: discovered_available,
            final_buffer_bytes: 0,
            sequence_count: 0,
            stage_flags: DISCOVERY_STAGE_STAGING_ACTIVE
                | DISCOVERY_STAGE_SOFTBUFFER_STOPPED
                | DISCOVERY_STAGE_FINALIZED,
        });
    }
    let staged_candidate = discovered_available;
    let mut stage_flags = DISCOVERY_STAGE_STAGING_ACTIVE;
    if snapshot.cpu_kernel_spinlock_flags != 0 { stage_flags |= DISCOVERY_STAGE_SPINLOCK_GUARDED; }
    let mut progressive_validated = staged_candidate;
    let mut sequence_count = 0u32;
    while sequence_count < 4 && progressive_validated > 1 {
        progressive_validated = progressive_validate_staged_size(progressive_validated);
        sequence_count = sequence_count.saturating_add(1);
    }
    if sequence_count > 0 { stage_flags |= DISCOVERY_STAGE_PROGRESSIVE_VALIDATED; }
    let ppm = soft_overhead_ppm.min(900_000) as u64;
    let soft_overhead_bytes = progressive_validated.saturating_mul(ppm).saturating_div(1_000_000);
    let final_buffer_bytes = progressive_validated.saturating_sub(soft_overhead_bytes);
    stage_flags |= DISCOVERY_STAGE_SOFT_OVERHEAD_APPLIED;
    stage_flags |= DISCOVERY_STAGE_FINALIZED;
    Some(BufferStagingProfile {
        discovered_total_bytes: discovered_total,
        discovered_available_bytes: discovered_available,
        staged_candidate_bytes: staged_candidate,
        progressive_validated_bytes: progressive_validated,
        soft_overhead_bytes,
        final_buffer_bytes,
        sequence_count,
        stage_flags,
    })
}

pub fn probe_hardware_buffer_staging_profile(
    backend: ComputeBackend,
) -> Option<BufferStagingProfile> {
    let snapshot = run_hardware_probe()?;
    match backend {
        ComputeBackend::Cpu => build_cpu_buffer_staging_profile(&snapshot, get_hardware_headroom_ppm()),
        ComputeBackend::Gpu => build_buffer_staging_profile(snapshot.gpu, get_hardware_headroom_ppm()),
        ComputeBackend::Tpu => build_buffer_staging_profile(snapshot.tpu, get_hardware_headroom_ppm()),
        ComputeBackend::Lpu => build_buffer_staging_profile(snapshot.lpu, get_hardware_headroom_ppm()),
    }
}

pub fn request_buffer_with_hardware_approval(
    backend: ComputeBackend,
    desired_bytes: usize,
    max_attempts: u32,
) -> Option<usize> {
    if desired_bytes == 0 { return None; }
    let mut attempt = 0u32;
    let mut candidate = desired_bytes as u64;
    while attempt < max_attempts && candidate > 0 {
        attempt = attempt.saturating_add(1);
        let snapshot = match run_hardware_probe() {
            Some(s) => s,
            None => {
                set_softbuffer_runtime_flags(
                    SOFTBUFFER_RUNTIME_FLAG_CPU_STOPPED
                        | SOFTBUFFER_RUNTIME_FLAG_GPU_STOPPED
                        | SOFTBUFFER_RUNTIME_FLAG_TPU_STOPPED
                        | SOFTBUFFER_RUNTIME_FLAG_LPU_STOPPED
                        | SOFTBUFFER_RUNTIME_FLAG_SNAPSHOT_MISSING,
                );
                for _ in 0..256 { spin_loop(); }
                candidate /= 2;
                continue;
            }
        };
        if backend != ComputeBackend::Cpu && !backend_present_from_detected_hardware(backend) {
            candidate /= 2;
            continue;
        }
        let staging_opt = match backend {
            ComputeBackend::Cpu => build_cpu_buffer_staging_profile(&snapshot, get_hardware_headroom_ppm()),
            ComputeBackend::Gpu => build_buffer_staging_profile(snapshot.gpu, get_hardware_headroom_ppm()),
            ComputeBackend::Tpu => build_buffer_staging_profile(snapshot.tpu, get_hardware_headroom_ppm()),
            ComputeBackend::Lpu => build_buffer_staging_profile(snapshot.lpu, get_hardware_headroom_ppm()),
        };
        if let Some(staging) = staging_opt {
            let quant_present = match backend {
                ComputeBackend::Cpu => snapshot.cpu_kernel_spinlock_flags & KERNEL_SPINLOCK_FLAG_QUANTIZATION != 0,
                ComputeBackend::Gpu => snapshot.gpu.kernel_spinlock_flags & KERNEL_SPINLOCK_FLAG_QUANTIZATION != 0,
                ComputeBackend::Tpu => snapshot.tpu.kernel_spinlock_flags & KERNEL_SPINLOCK_FLAG_QUANTIZATION != 0,
                ComputeBackend::Lpu => snapshot.lpu.kernel_spinlock_flags & KERNEL_SPINLOCK_FLAG_QUANTIZATION != 0,
            };
            if !quant_present {
                candidate = progressive_validate_staged_size(candidate);
                continue;
            }
            if staging.final_buffer_bytes > 0 {
                let approved = (candidate as usize).min(staging.final_buffer_bytes as usize);
                return Some(approved);
            } else {
                candidate /= 2;
                continue;
            }
        } else {
            candidate /= 2;
            continue;
        }
    }
    None
}

fn backend_detected_total_memory_auto(backend: ComputeBackend) -> usize {
    let Some(snapshot) = run_hardware_probe() else { return 0; };
    match backend {
        ComputeBackend::Cpu => {
            let Some(staging) = build_cpu_buffer_staging_profile(&snapshot, get_hardware_headroom_ppm()) else { return 0; };
            staging.discovered_total_bytes as usize
        }
        ComputeBackend::Gpu => snapshot.gpu.total_memory_bytes as usize,
        ComputeBackend::Tpu => snapshot.tpu.total_memory_bytes as usize,
        ComputeBackend::Lpu => snapshot.lpu.total_memory_bytes as usize,
    }
}

fn backend_detected_available_memory_auto(backend: ComputeBackend) -> usize {
    let Some(snapshot) = run_hardware_probe() else { return 0; };
    match backend {
        ComputeBackend::Cpu => {
            let Some(staging) = build_cpu_buffer_staging_profile(&snapshot, get_hardware_headroom_ppm()) else { return 0; };
            staging.final_buffer_bytes as usize
        }
        ComputeBackend::Gpu => snapshot.gpu.available_memory_bytes as usize,
        ComputeBackend::Tpu => snapshot.tpu.available_memory_bytes as usize,
        ComputeBackend::Lpu => snapshot.lpu.available_memory_bytes as usize,
    }
}

fn backend_detected_sustained_flops_auto(backend: ComputeBackend) -> usize {
    let Some(snapshot) = run_hardware_probe() else { return 0; };
    match backend {
        ComputeBackend::Cpu => 0,
        ComputeBackend::Gpu => snapshot.gpu.sustained_flops_per_second as usize,
        ComputeBackend::Tpu => snapshot.tpu.sustained_flops_per_second as usize,
        ComputeBackend::Lpu => snapshot.lpu.sustained_flops_per_second as usize,
    }
}

pub(super) fn backend_detected_sustained_flops_per_second(backend: ComputeBackend) -> usize {
    let raw = match backend {
        ComputeBackend::Cpu => 0,
        ComputeBackend::Gpu => GPU_SUSTAINED_FLOPS_PER_SECOND.load(Ordering::SeqCst),
        ComputeBackend::Tpu => TPU_SUSTAINED_FLOPS_PER_SECOND.load(Ordering::SeqCst),
        ComputeBackend::Lpu => LPU_SUSTAINED_FLOPS_PER_SECOND.load(Ordering::SeqCst),
    };
    let base = if raw == 0 { backend_detected_sustained_flops_auto(backend) } else { raw };
    apply_headroom(base, HARDWARE_HEADROOM_PPM.load(Ordering::SeqCst))
}

pub(super) fn backend_detected_total_memory_bytes(backend: ComputeBackend) -> usize {
    let raw = match backend {
        ComputeBackend::Cpu => 0,
        ComputeBackend::Gpu => GPU_TOTAL_MEMORY_BYTES.load(Ordering::SeqCst),
        ComputeBackend::Tpu => TPU_TOTAL_MEMORY_BYTES.load(Ordering::SeqCst),
        ComputeBackend::Lpu => LPU_TOTAL_MEMORY_BYTES.load(Ordering::SeqCst),
    };
    if raw == 0 { backend_detected_total_memory_auto(backend) } else { raw }
}

pub(super) fn backend_detected_available_memory_bytes(backend: ComputeBackend) -> usize {
    let raw = match backend {
        ComputeBackend::Cpu => 0,
        ComputeBackend::Gpu => GPU_AVAILABLE_MEMORY_BYTES.load(Ordering::SeqCst),
        ComputeBackend::Tpu => TPU_AVAILABLE_MEMORY_BYTES.load(Ordering::SeqCst),
        ComputeBackend::Lpu => LPU_AVAILABLE_MEMORY_BYTES.load(Ordering::SeqCst),
    };
    let total = backend_detected_total_memory_bytes(backend);
    if raw == 0 { backend_detected_available_memory_auto(backend).min(total) } else { raw.min(total) }
}

pub fn clear_backend_detected_profile(backend: ComputeBackend) {
    set_backend_detected_sustained_flops_per_second(backend, 0);
    set_backend_detected_memory(backend, 0, 0);
}

pub fn set_compute_backend(backend: ComputeBackend) {
    BACKEND.store(backend as u8, Ordering::SeqCst);
    if backend == ComputeBackend::Cpu {
        clear_hardware_probe();
    } else {
        register_default_hardware_probe_if_missing();
    }
    apply_backend_runtime_probe(backend);
    sync_backend_detected_profile(backend);
}

pub fn get_compute_backend() -> ComputeBackend {
    match BACKEND.load(Ordering::SeqCst) {
        1 => ComputeBackend::Gpu,
        2 => ComputeBackend::Tpu,
        3 => ComputeBackend::Lpu,
        _ => ComputeBackend::Cpu,
    }
}

pub fn set_hardware_contract_strict(strict: bool) {
    HARDWARE_CONTRACT_STRICT.store(if strict { 1 } else { 0 }, Ordering::SeqCst);
}

pub fn is_hardware_contract_strict() -> bool {
    HARDWARE_CONTRACT_STRICT.load(Ordering::SeqCst) != 0
}

pub fn set_contract_runtime_flags(flags: u32) {
    CONTRACT_RUNTIME_FLAGS.store((flags & CONTRACT_KNOWN_FLAGS_MASK) as usize, Ordering::SeqCst);
}

pub fn get_contract_runtime_flags() -> u32 {
    CONTRACT_RUNTIME_FLAGS.load(Ordering::SeqCst) as u32
}

pub fn set_contract_min_sustained_flops_per_second(min_flops_per_second: u128) {
    CONTRACT_MIN_SUSTAINED_FLOPS_PER_SECOND.store(saturating_u128_to_usize(min_flops_per_second), Ordering::SeqCst);
}

pub fn get_contract_min_sustained_flops_per_second() -> u128 {
    CONTRACT_MIN_SUSTAINED_FLOPS_PER_SECOND.load(Ordering::SeqCst) as u128
}

pub fn set_contract_min_available_memory_bytes(min_available_memory_bytes: u128) {
    CONTRACT_MIN_AVAILABLE_MEMORY_BYTES.store(saturating_u128_to_usize(min_available_memory_bytes), Ordering::SeqCst);
}

pub fn get_contract_min_available_memory_bytes() -> u128 {
    CONTRACT_MIN_AVAILABLE_MEMORY_BYTES.load(Ordering::SeqCst) as u128
}

pub fn set_backend_detected_sustained_flops_per_second(
    backend: ComputeBackend,
    sustained_flops_per_second: u128,
) {
    let value = saturating_u128_to_usize(sustained_flops_per_second);
    match backend {
        ComputeBackend::Cpu => {}
        ComputeBackend::Gpu => {
            GPU_SUSTAINED_FLOPS_PER_SECOND.store(value, Ordering::SeqCst);
        }
        ComputeBackend::Tpu => {
            TPU_SUSTAINED_FLOPS_PER_SECOND.store(value, Ordering::SeqCst);
        }
        ComputeBackend::Lpu => {
            LPU_SUSTAINED_FLOPS_PER_SECOND.store(value, Ordering::SeqCst);
        }
    }
}

pub fn set_backend_detected_memory(
    backend: ComputeBackend,
    total_memory_bytes: u128,
    available_memory_bytes: u128,
) {
    let total = saturating_u128_to_usize(total_memory_bytes);
    let available = saturating_u128_to_usize(available_memory_bytes).min(total);
    match backend {
        ComputeBackend::Cpu => {}
        ComputeBackend::Gpu => {
            GPU_TOTAL_MEMORY_BYTES.store(total, Ordering::SeqCst);
            GPU_AVAILABLE_MEMORY_BYTES.store(available, Ordering::SeqCst);
        }
        ComputeBackend::Tpu => {
            TPU_TOTAL_MEMORY_BYTES.store(total, Ordering::SeqCst);
            TPU_AVAILABLE_MEMORY_BYTES.store(available, Ordering::SeqCst);
        }
        ComputeBackend::Lpu => {
            LPU_TOTAL_MEMORY_BYTES.store(total, Ordering::SeqCst);
            LPU_AVAILABLE_MEMORY_BYTES.store(available, Ordering::SeqCst);
        }
    }
}

pub fn sync_detected_hardware_profile_to_contract() -> bool {
    let Some(snapshot) = run_hardware_probe() else { return false; };
    set_backend_detected_sustained_flops_per_second(
        ComputeBackend::Gpu,
        snapshot.gpu.sustained_flops_per_second as u128,
    );
    set_backend_detected_memory(
        ComputeBackend::Gpu,
        snapshot.gpu.total_memory_bytes as u128,
        snapshot.gpu.available_memory_bytes as u128,
    );
    set_backend_detected_sustained_flops_per_second(
        ComputeBackend::Tpu,
        snapshot.tpu.sustained_flops_per_second as u128,
    );
    set_backend_detected_memory(
        ComputeBackend::Tpu,
        snapshot.tpu.total_memory_bytes as u128,
        snapshot.tpu.available_memory_bytes as u128,
    );
    set_backend_detected_sustained_flops_per_second(
        ComputeBackend::Lpu,
        snapshot.lpu.sustained_flops_per_second as u128,
    );
    set_backend_detected_memory(
        ComputeBackend::Lpu,
        snapshot.lpu.total_memory_bytes as u128,
        snapshot.lpu.available_memory_bytes as u128,
    );
    true
}

pub fn sync_backend_detected_profile(backend: ComputeBackend) -> bool {
    match backend {
        ComputeBackend::Cpu => true,
        ComputeBackend::Gpu => {
            if get_gpu_contract_opcode_mask() == 0 {
                set_gpu_contract_opcode_mask(ALL_OPCODE_MASK);
            }
            backend_compute_present_from_detected_hardware(backend)
        }
        ComputeBackend::Tpu => {
            if get_tpu_contract_opcode_mask() == 0 {
                set_tpu_contract_opcode_mask(ALL_OPCODE_MASK);
            }
            backend_compute_present_from_detected_hardware(backend)
        }
        ComputeBackend::Lpu => {
            if get_lpu_contract_opcode_mask() == 0 {
                set_lpu_contract_opcode_mask(ALL_OPCODE_MASK);
            }
            backend_compute_present_from_detected_hardware(backend)
        }
    }
}

pub fn backend_detected_flop_abstraction(backend: ComputeBackend) -> Option<FlopAbstraction> {
    let flops = backend_detected_sustained_flops_per_second(backend) as u128;
    if flops == 0 { None } else { Some(build_flop_abstraction(flops)) }
}

pub fn active_backend_detected_flop_abstraction() -> Option<FlopAbstraction> {
    backend_detected_flop_abstraction(get_compute_backend())
}

pub fn backend_detected_ram_abstraction(backend: ComputeBackend) -> Option<RamAbstraction> {
    let total = backend_detected_total_memory_bytes(backend) as u128;
    let available = backend_detected_available_memory_bytes(backend) as u128;
    if total == 0 { None } else { Some(build_ram_abstraction(total, available)) }
}

pub fn active_backend_detected_ram_abstraction() -> Option<RamAbstraction> {
    backend_detected_ram_abstraction(get_compute_backend())
}

pub fn backend_supported_request_flags(backend: ComputeBackend) -> u32 {
    match backend {
        ComputeBackend::Cpu => 0,
        ComputeBackend::Gpu => CONTRACT_KNOWN_FLAGS_MASK,
        ComputeBackend::Tpu => CONTRACT_KNOWN_FLAGS_MASK,
        ComputeBackend::Lpu => CONTRACT_KNOWN_FLAGS_MASK,
    }
}

pub fn register_contract_reject_handler(handler: ContractRejectHandler, user_ctx: usize) {
    CONTRACT_REJECT_USER_CTX.store(user_ctx, Ordering::SeqCst);
    CONTRACT_REJECT_HANDLER.store(handler as usize, Ordering::SeqCst);
}

pub fn clear_contract_reject_handler() {
    CONTRACT_REJECT_HANDLER.store(0, Ordering::SeqCst);
    CONTRACT_REJECT_USER_CTX.store(0, Ordering::SeqCst);
}

pub fn clear_last_contract_reject() {
    LAST_CONTRACT_REJECT_REASON.store(0, Ordering::SeqCst);
    LAST_CONTRACT_REJECT_BACKEND.store(ComputeBackend::Cpu as u8, Ordering::SeqCst);
    LAST_CONTRACT_REJECT_OPCODE.store(0, Ordering::SeqCst);
}

pub fn last_contract_reject_reason() -> Option<ContractRejectReason> {
    let raw = LAST_CONTRACT_REJECT_REASON.load(Ordering::SeqCst) as u32;
    contract_reject_reason_from_raw(raw)
}

pub fn last_contract_reject_backend() -> Option<ComputeBackend> {
    let has_reason = LAST_CONTRACT_REJECT_REASON.load(Ordering::SeqCst) != 0;
    if !has_reason { return None; }
    backend_from_raw(LAST_CONTRACT_REJECT_BACKEND.load(Ordering::SeqCst))
}

pub fn last_contract_reject_opcode() -> Option<GpuOpCode> {
    let has_reason = LAST_CONTRACT_REJECT_REASON.load(Ordering::SeqCst) != 0;
    if !has_reason { return None; }
    let raw = LAST_CONTRACT_REJECT_OPCODE.load(Ordering::SeqCst) as u32;
    opcode_from_raw(raw)
}

pub fn set_gpu_contract_opcode_mask(opcode_mask: u32) {
    GPU_OPCODE_MASK.store((opcode_mask & ALL_OPCODE_MASK) as usize, Ordering::SeqCst);
}

pub fn set_tpu_contract_opcode_mask(opcode_mask: u32) {
    TPU_OPCODE_MASK.store((opcode_mask & ALL_OPCODE_MASK) as usize, Ordering::SeqCst);
}

pub fn set_lpu_contract_opcode_mask(opcode_mask: u32) {
    LPU_OPCODE_MASK.store((opcode_mask & ALL_OPCODE_MASK) as usize, Ordering::SeqCst);
}

pub fn get_gpu_contract_opcode_mask() -> u32 {
    GPU_OPCODE_MASK.load(Ordering::SeqCst) as u32
}

pub fn get_tpu_contract_opcode_mask() -> u32 {
    TPU_OPCODE_MASK.load(Ordering::SeqCst) as u32
}

pub fn get_lpu_contract_opcode_mask() -> u32 {
    LPU_OPCODE_MASK.load(Ordering::SeqCst) as u32
}

pub fn register_gpu_command_handler_with_capabilities(
    handler: GpuCommandHandler, user_ctx: usize, opcode_mask: u32,
) {
    set_gpu_contract_opcode_mask(opcode_mask);
    register_gpu_command_handler(handler, user_ctx);
}

pub fn register_gpu_command_handler_with_hardware_profile(
    handler: GpuCommandHandler, user_ctx: usize, opcode_mask: u32,
    sustained_flops_per_second: u128, total_memory_bytes: u128, available_memory_bytes: u128,
) {
    set_backend_detected_sustained_flops_per_second(ComputeBackend::Gpu, sustained_flops_per_second);
    set_backend_detected_memory(ComputeBackend::Gpu, total_memory_bytes, available_memory_bytes);
    register_gpu_command_handler_with_capabilities(handler, user_ctx, opcode_mask);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn register_tpu_command_handler_with_capabilities(
    handler: TpuCommandHandler, user_ctx: usize, opcode_mask: u32,
) {
    set_tpu_contract_opcode_mask(opcode_mask);
    register_tpu_command_handler(handler, user_ctx);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn register_tpu_command_handler_with_hardware_profile(
    handler: TpuCommandHandler, user_ctx: usize, opcode_mask: u32,
    sustained_flops_per_second: u128, total_memory_bytes: u128, available_memory_bytes: u128,
) {
    set_backend_detected_sustained_flops_per_second(ComputeBackend::Tpu, sustained_flops_per_second);
    set_backend_detected_memory(ComputeBackend::Tpu, total_memory_bytes, available_memory_bytes);
    register_tpu_command_handler_with_capabilities(handler, user_ctx, opcode_mask);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn register_lpu_command_handler_with_capabilities(
    handler: LpuCommandHandler, user_ctx: usize, opcode_mask: u32,
) {
    set_lpu_contract_opcode_mask(opcode_mask);
    register_lpu_command_handler(handler, user_ctx);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn register_lpu_command_handler_with_hardware_profile(
    handler: LpuCommandHandler, user_ctx: usize, opcode_mask: u32,
    sustained_flops_per_second: u128, total_memory_bytes: u128, available_memory_bytes: u128,
) {
    set_backend_detected_sustained_flops_per_second(ComputeBackend::Lpu, sustained_flops_per_second);
    set_backend_detected_memory(ComputeBackend::Lpu, total_memory_bytes, available_memory_bytes);
    register_lpu_command_handler_with_capabilities(handler, user_ctx, opcode_mask);
}

pub(super) fn contract_dispatch_reject(
    reason: ContractRejectReason,
    backend: ComputeBackend,
    opcode: GpuOpCode,
) -> bool {
    LAST_CONTRACT_REJECT_REASON.store(reason as usize, Ordering::SeqCst);
    LAST_CONTRACT_REJECT_BACKEND.store(backend as u8, Ordering::SeqCst);
    LAST_CONTRACT_REJECT_OPCODE.store(opcode as usize, Ordering::SeqCst);
    if is_hardware_contract_strict() {
        let handler_ptr = CONTRACT_REJECT_HANDLER.load(Ordering::SeqCst);
        if let Some(handler) = decode_contract_reject_handler(handler_ptr) {
            let user_ctx = CONTRACT_REJECT_USER_CTX.load(Ordering::SeqCst);
            handler(reason, backend, opcode as u32, user_ctx);
            return false;
        }
    }
    false
}

pub fn register_gpu_kernel_f32(handler: GpuKernelF32) {
    GPU_KERNEL_F32.store(handler as usize, Ordering::SeqCst);
    decode_gpu_kernel_f32(GPU_KERNEL_F32.load(Ordering::SeqCst));
}

pub fn register_gpu_kernel_f64(handler: GpuKernelF64) {
    GPU_KERNEL_F64.store(handler as usize, Ordering::SeqCst);
    decode_gpu_kernel_f64(GPU_KERNEL_F64.load(Ordering::SeqCst));
}

pub fn clear_gpu_kernel_f32() { GPU_KERNEL_F32.store(0, Ordering::SeqCst); }
pub fn clear_gpu_kernel_f64() { GPU_KERNEL_F64.store(0, Ordering::SeqCst); }

pub fn register_cpu_kernel_f32(handler: GpuKernelF32) {
    CPU_KERNEL_F32.store(handler as usize, Ordering::SeqCst);
}

pub fn register_cpu_kernel_f64(handler: GpuKernelF64) {
    CPU_KERNEL_F64.store(handler as usize, Ordering::SeqCst);
}

pub fn clear_cpu_kernel_f32() { CPU_KERNEL_F32.store(0, Ordering::SeqCst); }
pub fn clear_cpu_kernel_f64() { CPU_KERNEL_F64.store(0, Ordering::SeqCst); }

pub fn register_gpu_command_handler(handler: GpuCommandHandler, user_ctx: usize) {
    GPU_COMMAND_USER_CTX.store(user_ctx, Ordering::SeqCst);
    GPU_COMMAND_HANDLER.store(handler as usize, Ordering::SeqCst);
}

pub fn clear_gpu_command_handler() {
    GPU_COMMAND_HANDLER.store(0, Ordering::SeqCst);
    GPU_COMMAND_USER_CTX.store(0, Ordering::SeqCst);
    GPU_OPCODE_MASK.store(0, Ordering::SeqCst);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn register_tpu_command_handler(handler: TpuCommandHandler, user_ctx: usize) {
    TPU_COMMAND_USER_CTX.store(user_ctx, Ordering::SeqCst);
    TPU_COMMAND_HANDLER.store(handler as usize, Ordering::SeqCst);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn clear_tpu_command_handler() {
    TPU_COMMAND_HANDLER.store(0, Ordering::SeqCst);
    TPU_COMMAND_USER_CTX.store(0, Ordering::SeqCst);
    TPU_OPCODE_MASK.store(0, Ordering::SeqCst);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn register_lpu_command_handler(handler: LpuCommandHandler, user_ctx: usize) {
    LPU_COMMAND_USER_CTX.store(user_ctx, Ordering::SeqCst);
    LPU_COMMAND_HANDLER.store(handler as usize, Ordering::SeqCst);
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub fn clear_lpu_command_handler() {
    LPU_COMMAND_HANDLER.store(0, Ordering::SeqCst);
    LPU_COMMAND_USER_CTX.store(0, Ordering::SeqCst);
    LPU_OPCODE_MASK.store(0, Ordering::SeqCst);
}

fn hash_payload(payload_ptr: *const u8, payload_len: usize) -> u32 {
    let mut hash: u32 = 0x811C9DC5;
    let mut i = 0usize;
    while i < payload_len {
        let b = unsafe { *payload_ptr.add(i) };
        hash ^= b as u32;
        hash = hash.wrapping_mul(0x01000193);
        i += 1;
    }
    hash
}

pub fn validate_contract_request_for_backend(
    backend: ComputeBackend,
    request: &GpuContractRequest,
) -> GpuDispatchStatus {
    let Some(caps) = probe_backend_contract(backend) else { return GpuDispatchStatus::BadContract; };
    let expected_magic = match backend {
        ComputeBackend::Cpu => return GpuDispatchStatus::BadContract,
        ComputeBackend::Gpu => GPU_CONTRACT_MAGIC,
        ComputeBackend::Tpu => TPU_CONTRACT_MAGIC,
        ComputeBackend::Lpu => LPU_CONTRACT_MAGIC,
    };
    if request.header.magic != expected_magic
        || !is_contract_abi_compatible(backend, request.header.abi_version)
    {
        return GpuDispatchStatus::BadContract;
    }
    let supported_flags = backend_supported_request_flags(backend);
    if request.header.flags & !supported_flags != 0 { return GpuDispatchStatus::BadContract; }
    if caps.total_memory_bytes == 0 || caps.available_memory_bytes > caps.total_memory_bytes {
        return GpuDispatchStatus::BadContract;
    }
    let min_flops = get_contract_min_sustained_flops_per_second() as u64;
    if min_flops > 0 && caps.sustained_flops_per_second < min_flops { return GpuDispatchStatus::NotSupported; }
    let min_available_memory = get_contract_min_available_memory_bytes() as u64;
    if min_available_memory > 0 && caps.available_memory_bytes < min_available_memory { return GpuDispatchStatus::NotSupported; }
    let Some(opcode) = opcode_from_raw(request.header.opcode) else { return GpuDispatchStatus::BadContract; };
    if !backend_supports_opcode(backend, opcode) { return GpuDispatchStatus::NotSupported; }
    let epl = expected_payload_len(opcode);
    if request.header.payload_len as usize != epl { return GpuDispatchStatus::BadPayload; }
    let payload_len = request.header.payload_len as usize;
    if payload_len > 0 && request.payload_ptr.is_null() { return GpuDispatchStatus::BadPayload; }
    let actual_hash = if payload_len == 0 { 0x811C9DC5u32 } else { hash_payload(request.payload_ptr, payload_len) };
    if request.header.payload_hash != actual_hash { return GpuDispatchStatus::BadPayload; }
    GpuDispatchStatus::Ok
}

fn active_contract_info() -> Option<(u32, u16, usize, usize)> {
    match get_compute_backend() {
        ComputeBackend::Cpu => None,
        ComputeBackend::Gpu => Some((
            GPU_CONTRACT_MAGIC,
            GPU_CONTRACT_ABI_VERSION,
            GPU_COMMAND_HANDLER.load(Ordering::SeqCst),
            GPU_COMMAND_USER_CTX.load(Ordering::SeqCst),
        )),
        ComputeBackend::Tpu => Some((
            TPU_CONTRACT_MAGIC,
            TPU_CONTRACT_ABI_VERSION,
            TPU_COMMAND_HANDLER.load(Ordering::SeqCst),
            TPU_COMMAND_USER_CTX.load(Ordering::SeqCst),
        )),
        ComputeBackend::Lpu => Some((
            LPU_CONTRACT_MAGIC,
            LPU_CONTRACT_ABI_VERSION,
            LPU_COMMAND_HANDLER.load(Ordering::SeqCst),
            LPU_COMMAND_USER_CTX.load(Ordering::SeqCst),
        )),
    }
}

pub(super) fn dispatch_contract(opcode: GpuOpCode, payload_ptr: *const u8, payload_len: usize) -> bool {
    let backend = get_compute_backend();
    let Some(caps) = active_backend_contract_capabilities() else {
        return contract_dispatch_reject(ContractRejectReason::NoActiveBackendContract, backend, opcode);
    };
    if (caps.opcode_mask & opcode_bit(opcode)) == 0 {
        return contract_dispatch_reject(ContractRejectReason::OpcodeNotSupported, backend, opcode);
    }
    if payload_len != expected_payload_len(opcode) {
        return contract_dispatch_reject(ContractRejectReason::PayloadSizeMismatch, backend, opcode);
    }
    if payload_len > 0 && payload_ptr.is_null() {
        return contract_dispatch_reject(ContractRejectReason::NullPayloadPointer, backend, opcode);
    }
    let runtime_flags = get_contract_runtime_flags();
    let supported_flags = backend_supported_request_flags(backend);
    if runtime_flags & !supported_flags != 0 {
        return contract_dispatch_reject(ContractRejectReason::UnsupportedRequestFlags, backend, opcode);
    }
    let min_flops = get_contract_min_sustained_flops_per_second() as u64;
    if min_flops > 0 && caps.sustained_flops_per_second < min_flops {
        return contract_dispatch_reject(ContractRejectReason::InsufficientSustainedFlops, backend, opcode);
    }
    let min_available_memory = get_contract_min_available_memory_bytes() as u64;
    if min_available_memory > 0 && caps.available_memory_bytes < min_available_memory {
        return contract_dispatch_reject(ContractRejectReason::InsufficientAvailableMemory, backend, opcode);
    }
    let Some((magic, abi_version, handler_ptr, user_ctx)) = active_contract_info() else {
        return contract_dispatch_reject(ContractRejectReason::NoActiveContractInfo, backend, opcode);
    };
    let Some(handler) = decode_gpu_command_handler(handler_ptr) else {
        return contract_dispatch_reject(ContractRejectReason::NoRegisteredHandler, backend, opcode);
    };
    let header = GpuContractHeader {
        magic, abi_version, reserved: 0,
        opcode: opcode as u32,
        payload_len: payload_len as u32,
        payload_hash: hash_payload(payload_ptr, payload_len),
        flags: runtime_flags,
    };
    let request = GpuContractRequest { header, payload_ptr };
    let status = handler(&request as *const GpuContractRequest, user_ctx);
    if status == GpuDispatchStatus::Ok as u32 { true }
    else { contract_dispatch_reject(ContractRejectReason::HandlerReturnedNonOk, backend, opcode) }
}

pub fn probe_backend_contract(backend: ComputeBackend) -> Option<BackendContractCapabilities> {
    if backend != ComputeBackend::Cpu {
        sync_backend_detected_profile(backend);
    }
    if backend != ComputeBackend::Cpu && !backend_compute_present_from_detected_hardware(backend) {
        return None;
    }
    let (magic, abi_version, opcode_mask, preferred_alignment) = match backend {
        ComputeBackend::Cpu => return None,
        ComputeBackend::Gpu => {
            let mask = get_gpu_contract_opcode_mask();
            (GPU_CONTRACT_MAGIC, GPU_CONTRACT_ABI_VERSION, if mask == 0 { ALL_OPCODE_MASK } else { mask }, 64)
        }
        ComputeBackend::Tpu => {
            let mask = get_tpu_contract_opcode_mask();
            (TPU_CONTRACT_MAGIC, TPU_CONTRACT_ABI_VERSION, if mask == 0 { ALL_OPCODE_MASK } else { mask }, 64)
        }
        ComputeBackend::Lpu => {
            let mask = get_lpu_contract_opcode_mask();
            (LPU_CONTRACT_MAGIC, LPU_CONTRACT_ABI_VERSION, if mask == 0 { ALL_OPCODE_MASK } else { mask }, 64)
        }
    };
    Some(BackendContractCapabilities {
        magic, abi_version, reserved: 0, opcode_mask,
        feature_flags: feature_flags_from_opcode_mask(opcode_mask),
        max_batch: u32::MAX, max_elements: u32::MAX, preferred_alignment,
        sustained_flops_per_second: backend_detected_sustained_flops_per_second(backend) as u64,
        total_memory_bytes: backend_detected_total_memory_bytes(backend) as u64,
        available_memory_bytes: backend_detected_available_memory_bytes(backend) as u64,
    })
}

pub fn active_backend_contract_capabilities() -> Option<BackendContractCapabilities> {
    probe_backend_contract(get_compute_backend())
}

pub fn backend_supports_opcode(backend: ComputeBackend, opcode: GpuOpCode) -> bool {
    probe_backend_contract(backend)
        .map(|caps| (caps.opcode_mask & opcode_bit(opcode)) != 0)
        .unwrap_or(false)
}
