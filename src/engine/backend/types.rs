use crate::base::activations::ActivationKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ComputeBackend {
    Cpu = 0,
    Gpu = 1,
    Tpu = 2,
    Lpu = 3,
}

pub const GPU_CONTRACT_MAGIC: u32 = 0x4750_5543;
pub const GPU_CONTRACT_ABI_VERSION: u16 = 1;
pub const TPU_CONTRACT_MAGIC: u32 = 0x5450_5543;
pub const TPU_CONTRACT_ABI_VERSION: u16 = 1;
pub const LPU_CONTRACT_MAGIC: u32 = 0x4C50_5543;
pub const LPU_CONTRACT_ABI_VERSION: u16 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuContractHeader {
    pub magic: u32,
    pub abi_version: u16,
    pub reserved: u16,
    pub opcode: u32,
    pub payload_len: u32,
    pub payload_hash: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuContractRequest {
    pub header: GpuContractHeader,
    pub payload_ptr: *const u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuOpCode {
    F32 = 1,
    F64 = 2,
    SoftmaxF32 = 3,
    SoftmaxF64 = 4,
    LayerNormF32 = 5,
    LayerNormF64 = 6,
    RmsNormF32 = 7,
    RmsNormF64 = 8,
    AttentionF32 = 9,
    AttentionF64 = 10,
    QuantizeI8F32 = 11,
    QuantizeI8F64 = 12,
    DequantizeI8F32 = 13,
    DequantizeI8F64 = 14,
    SgdF32 = 15,
    SgdF64 = 16,
    AdamwF32 = 17,
    AdamwF64 = 18,
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type TpuOpCode = GpuOpCode;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type LpuOpCode = GpuOpCode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuDispatchStatus {
    Ok = 0,
    NotSupported = 1,
    Failed = 2,
    BadContract = 3,
    BadPayload = 4,
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type TpuDispatchStatus = GpuDispatchStatus;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type LpuDispatchStatus = GpuDispatchStatus;

pub const CONTRACT_FEATURE_F32: u32 = 1 << 0;
pub const CONTRACT_FEATURE_F64: u32 = 1 << 1;
pub const CONTRACT_FEATURE_INT8: u32 = 1 << 2;
pub const CONTRACT_FEATURE_BATCH: u32 = 1 << 3;

pub const HOST_OS_FLAG_UNKNOWN: u32 = 0;
pub const HOST_OS_FLAG_LINUX: u32 = 1;
pub const HOST_OS_FLAG_WINDOWS: u32 = 2;
pub const HOST_OS_FLAG_MACOS: u32 = 3;

pub const HARDWARE_FLAG_CPU: u32 = 1 << 0;
pub const HARDWARE_FLAG_GPU: u32 = 1 << 1;
pub const HARDWARE_FLAG_TPU: u32 = 1 << 2;
pub const HARDWARE_FLAG_LPU: u32 = 1 << 3;

pub const KERNEL_SPINLOCK_FLAG: u64 = 1 << 0;
pub const KERNEL_SPINLOCK_FLAG_SOFTMAX: u64 = 1 << 1;
pub const KERNEL_SPINLOCK_FLAG_LAYER_NORM: u64 = 1 << 2;
pub const KERNEL_SPINLOCK_FLAG_RMS_NORM: u64 = 1 << 3;
pub const KERNEL_SPINLOCK_FLAG_ATTENTION: u64 = 1 << 4;
pub const KERNEL_SPINLOCK_FLAG_QUANTIZATION: u64 = 1 << 5;
pub const KERNEL_SPINLOCK_FLAG_OPTIMIZER: u64 = 1 << 6;
pub const KERNEL_SPINLOCK_FLAG_GLOBAL: u64 = 1 << 7;
pub const KERNEL_SPINLOCK_FLAG_OS_LINUX_HINT: u64 = 1 << 60;
pub const KERNEL_SPINLOCK_FLAG_OS_WINDOWS_HINT: u64 = 1 << 61;
pub const KERNEL_SPINLOCK_FLAG_OS_MACOS_HINT: u64 = 1 << 62;

pub const HARDWARE_TIMING_FLAG_CLOCK_HZ_VALID: u64 = 1 << 0;
pub const HARDWARE_TIMING_FLAG_CYCLE_COUNTER_VALID: u64 = 1 << 1;
pub const HARDWARE_TIMING_FLAG_NOMINAL_CYCLES_VALID: u64 = 1 << 2;
pub const HARDWARE_TIMING_FLAG_OBSERVED_CYCLES_VALID: u64 = 1 << 3;
pub const HARDWARE_PROBE_FLAG_HANDLER_REGISTERED: u32 = 1 << 0;
pub const HARDWARE_PROBE_FLAG_SNAPSHOT_AVAILABLE: u32 = 1 << 1;
pub const DISCOVERY_STAGE_SPINLOCK_GUARDED: u32 = 1 << 0;
pub const DISCOVERY_STAGE_STAGING_ACTIVE: u32 = 1 << 1;
pub const DISCOVERY_STAGE_PROGRESSIVE_VALIDATED: u32 = 1 << 2;
pub const DISCOVERY_STAGE_SOFT_OVERHEAD_APPLIED: u32 = 1 << 3;
pub const DISCOVERY_STAGE_FINALIZED: u32 = 1 << 4;
pub const DISCOVERY_STAGE_SOFTBUFFER_STOPPED: u32 = 1 << 5;

pub const SOFTBUFFER_RUNTIME_FLAG_CPU_STOPPED: u32 = 1 << 0;
pub const SOFTBUFFER_RUNTIME_FLAG_GPU_STOPPED: u32 = 1 << 1;
pub const SOFTBUFFER_RUNTIME_FLAG_TPU_STOPPED: u32 = 1 << 2;
pub const SOFTBUFFER_RUNTIME_FLAG_LPU_STOPPED: u32 = 1 << 3;
pub const SOFTBUFFER_RUNTIME_FLAG_SNAPSHOT_MISSING: u32 = 1 << 31;

pub const CONTRACT_FLAG_REQUIRE_FINITE_INPUTS: u32 = 1 << 0;
pub const CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT: u32 = 1 << 1;
pub const CONTRACT_FLAG_DETERMINISTIC_MATH: u32 = 1 << 2;
pub const CONTRACT_KNOWN_FLAGS_MASK: u32 = CONTRACT_FLAG_REQUIRE_FINITE_INPUTS
    | CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT
    | CONTRACT_FLAG_DETERMINISTIC_MATH;

pub const FLOPS_SCALE_GIGA: u128 = 1_000_000_000;
pub const FLOPS_SCALE_TERA: u128 = 1_000_000_000_000;
pub const FLOPS_SCALE_LARGE: u128 = 1_000_000_000_000_000;

pub const DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND: u64 = 1_000_000_000_000;
pub const DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND: u64 = 2_000_000_000_000;
pub const DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND: u64 = 250_000_000_000;

pub const DEFAULT_GPU_TOTAL_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
pub const DEFAULT_TPU_TOTAL_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
pub const DEFAULT_LPU_TOTAL_MEMORY_BYTES: u64 = 4 * 1024 * 1024 * 1024;
pub const DEFAULT_HARDWARE_HEADROOM_PPM: usize = 50_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlopAbstraction {
    pub flops_per_second: u128,
    pub gflops_x1000: u128,
    pub tflops_x1000: u128,
    pub lflops_x1000: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamAbstraction {
    pub total_bytes: u128,
    pub available_bytes: u128,
    pub used_bytes: u128,
    pub total_mib: u128,
    pub available_mib: u128,
    pub used_mib: u128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendTimingAbstraction {
    pub clock_hz: u64,
    pub cycle_counter_hz: u64,
    pub nominal_cycles_per_kernel: u64,
    pub observed_cycles_per_kernel: u64,
    pub timing_flags: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferStagingProfile {
    pub discovered_total_bytes: u64,
    pub discovered_available_bytes: u64,
    pub staged_candidate_bytes: u64,
    pub progressive_validated_bytes: u64,
    pub soft_overhead_bytes: u64,
    pub final_buffer_bytes: u64,
    pub sequence_count: u32,
    pub stage_flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HardwareComponentSnapshot {
    pub present: u8,
    pub reserved0: u8,
    pub reserved1: u8,
    pub reserved2: u8,
    pub device_count: u32,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
    pub sustained_flops_per_second: u64,
    pub kernel_spinlock_flags: u64,
    pub clock_hz: u64,
    pub cycle_counter_hz: u64,
    pub nominal_cycles_per_kernel: u64,
    pub observed_cycles_per_kernel: u64,
    pub timing_flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct HardwareDetectionSnapshot {
    pub cpu_logical_cores: u32,
    pub os_flags: u32,
    pub hardware_flags: u32,
    pub system_total_memory_bytes: u64,
    pub system_available_memory_bytes: u64,
    pub cpu_kernel_spinlock_flags: u64,
    pub cpu_clock_hz: u64,
    pub cpu_cycle_counter_hz: u64,
    pub cpu_nominal_cycles_per_kernel: u64,
    pub cpu_observed_cycles_per_kernel: u64,
    pub cpu_timing_flags: u64,
    pub gpu: HardwareComponentSnapshot,
    pub tpu: HardwareComponentSnapshot,
    pub lpu: HardwareComponentSnapshot,
}

pub type HardwareProbeFn =
    fn(out_snapshot: *mut HardwareDetectionSnapshot, user_ctx: usize) -> u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ContractRejectReason {
    NoActiveBackendContract = 1,
    OpcodeNotSupported = 2,
    PayloadSizeMismatch = 3,
    NullPayloadPointer = 4,
    NoActiveContractInfo = 5,
    NoRegisteredHandler = 6,
    HandlerReturnedNonOk = 7,
    UnsupportedRequestFlags = 8,
    InsufficientSustainedFlops = 9,
    InsufficientAvailableMemory = 10,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BackendContractCapabilities {
    pub magic: u32,
    pub abi_version: u16,
    pub reserved: u16,
    pub opcode_mask: u32,
    pub feature_flags: u32,
    pub max_batch: u32,
    pub max_elements: u32,
    pub preferred_alignment: u32,
    pub sustained_flops_per_second: u64,
    pub total_memory_bytes: u64,
    pub available_memory_bytes: u64,
}

pub type GpuCommandHandler =
    fn(request: *const GpuContractRequest, user_ctx: usize) -> u32;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type TpuCommandHandler =
    fn(request: *const GpuContractRequest, user_ctx: usize) -> u32;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type LpuCommandHandler =
    fn(request: *const GpuContractRequest, user_ctx: usize) -> u32;
pub type ContractRejectHandler = fn(
    reason: ContractRejectReason,
    backend: ComputeBackend,
    opcode: u32,
    user_ctx: usize,
);

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type TpuContractHeader = GpuContractHeader;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type TpuContractRequest = GpuContractRequest;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type LpuContractHeader = GpuContractHeader;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub type LpuContractRequest = GpuContractRequest;

pub type GpuKernelF32 = fn(args: KernelInvokeF32<'_>) -> bool;

pub type GpuKernelF64 = fn(args: KernelInvokeF64<'_>) -> bool;

#[repr(C)]
pub struct KernelCmdF32 {
    pub src_ptr: *const f32,
    pub src_len: usize,
    pub dst_ptr: *mut f32,
    pub dst_len: usize,
    pub batch_size: usize,
    pub stride: usize,
    pub in_size: usize,
    pub out_size: usize,
    pub weights_ptr: *const f32,
    pub weights_len: usize,
    pub biases_ptr: *const f32,
    pub biases_len: usize,
    pub activation: u8,
}

#[repr(C)]
pub struct KernelCmdF64 {
    pub src_ptr: *const f64,
    pub src_len: usize,
    pub dst_ptr: *mut f64,
    pub dst_len: usize,
    pub batch_size: usize,
    pub stride: usize,
    pub in_size: usize,
    pub out_size: usize,
    pub weights_ptr: *const f64,
    pub weights_len: usize,
    pub biases_ptr: *const f64,
    pub biases_len: usize,
    pub activation: u8,
}

#[repr(C)]
pub struct SoftmaxCmdF32 {
    pub logits_ptr: *const f32,
    pub out_ptr: *mut f32,
    pub len: usize,
}

#[repr(C)]
pub struct SoftmaxCmdF64 {
    pub logits_ptr: *const f64,
    pub out_ptr: *mut f64,
    pub len: usize,
}

#[repr(C)]
pub struct LayerNormCmdF32 {
    pub x_ptr: *mut f32,
    pub gamma_ptr: *const f32,
    pub beta_ptr: *const f32,
    pub len: usize,
    pub eps: f32,
}

#[repr(C)]
pub struct LayerNormCmdF64 {
    pub x_ptr: *mut f64,
    pub gamma_ptr: *const f64,
    pub beta_ptr: *const f64,
    pub len: usize,
    pub eps: f64,
}

#[repr(C)]
pub struct RmsNormCmdF32 {
    pub x_ptr: *mut f32,
    pub gamma_ptr: *const f32,
    pub len: usize,
    pub eps: f32,
}

#[repr(C)]
pub struct RmsNormCmdF64 {
    pub x_ptr: *mut f64,
    pub gamma_ptr: *const f64,
    pub len: usize,
    pub eps: f64,
}

#[repr(C)]
pub struct AttentionCmdF32 {
    pub q_ptr: *const f32,
    pub q_len_total: usize,
    pub k_ptr: *const f32,
    pub k_len_total: usize,
    pub v_ptr: *const f32,
    pub v_len_total: usize,
    pub out_ptr: *mut f32,
    pub out_len_total: usize,
    pub scratch_scores_ptr: *mut f32,
    pub scratch_scores_len: usize,
    pub q_len: usize,
    pub k_len: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub k_stride: usize,
    pub v_stride: usize,
    pub head_offset: usize,
    pub mask: u8,
}

#[repr(C)]
pub struct AttentionCmdF64 {
    pub q_ptr: *const f64,
    pub q_len_total: usize,
    pub k_ptr: *const f64,
    pub k_len_total: usize,
    pub v_ptr: *const f64,
    pub v_len_total: usize,
    pub out_ptr: *mut f64,
    pub out_len_total: usize,
    pub scratch_scores_ptr: *mut f64,
    pub scratch_scores_len: usize,
    pub q_len: usize,
    pub k_len: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub k_stride: usize,
    pub v_stride: usize,
    pub head_offset: usize,
    pub mask: u8,
}

#[repr(C)]
pub struct QuantizeI8CmdF32 {
    pub input_ptr: *const f32,
    pub input_len: usize,
    pub output_ptr: *mut i8,
    pub output_len: usize,
    pub scale_out_ptr: *mut f32,
}

#[repr(C)]
pub struct QuantizeI8CmdF64 {
    pub input_ptr: *const f64,
    pub input_len: usize,
    pub output_ptr: *mut i8,
    pub output_len: usize,
    pub scale_out_ptr: *mut f64,
}

#[repr(C)]
pub struct DequantizeI8CmdF32 {
    pub input_ptr: *const i8,
    pub input_len: usize,
    pub output_ptr: *mut f32,
    pub output_len: usize,
    pub scale: f32,
}

#[repr(C)]
pub struct DequantizeI8CmdF64 {
    pub input_ptr: *const i8,
    pub input_len: usize,
    pub output_ptr: *mut f64,
    pub output_len: usize,
    pub scale: f64,
}

#[repr(C)]
pub struct SgdCmdF32 {
    pub params_ptr: *mut f32,
    pub grads_ptr: *const f32,
    pub velocity_ptr: *mut f32,
    pub len: usize,
    pub learning_rate: f32,
    pub momentum: f32,
    pub nesterov: u8,
}

#[repr(C)]
pub struct SgdCmdF64 {
    pub params_ptr: *mut f64,
    pub grads_ptr: *const f64,
    pub velocity_ptr: *mut f64,
    pub len: usize,
    pub learning_rate: f64,
    pub momentum: f64,
    pub nesterov: u8,
}

#[repr(C)]
pub struct AdamwCmdF32 {
    pub params_ptr: *mut f32,
    pub grads_ptr: *const f32,
    pub m_ptr: *mut f32,
    pub v_ptr: *mut f32,
    pub len: usize,
    pub learning_rate: f32,
    pub step: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

#[repr(C)]
pub struct AdamwCmdF64 {
    pub params_ptr: *mut f64,
    pub grads_ptr: *const f64,
    pub m_ptr: *mut f64,
    pub v_ptr: *mut f64,
    pub len: usize,
    pub learning_rate: f64,
    pub step: u32,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

pub struct KernelInvokeF32<'a> {
    pub src: &'a [f32],
    pub dst: &'a mut [f32],
    pub batch_size: usize,
    pub stride: usize,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: &'a [f32],
    pub biases: &'a [f32],
    pub activation: ActivationKind,
}

pub struct KernelInvokeF64<'a> {
    pub src: &'a [f64],
    pub dst: &'a mut [f64],
    pub batch_size: usize,
    pub stride: usize,
    pub in_size: usize,
    pub out_size: usize,
    pub weights: &'a [f64],
    pub biases: &'a [f64],
    pub activation: ActivationKind,
}

pub struct AttentionInvoke<'a, T> {
    pub q: &'a [T],
    pub k: &'a [T],
    pub v: &'a [T],
    pub out: &'a mut [T],
    pub scratch_scores: &'a mut [T],
    pub q_len: usize,
    pub k_len: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub k_stride: usize,
    pub v_stride: usize,
    pub head_offset: usize,
    pub mask: u8,
}

pub struct AttentionInvokeF32<'a> {
    pub q: &'a [f32],
    pub k: &'a [f32],
    pub v: &'a [f32],
    pub out: &'a mut [f32],
    pub scratch_scores: &'a mut [f32],
    pub q_len: usize,
    pub k_len: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub k_stride: usize,
    pub v_stride: usize,
    pub head_offset: usize,
    pub mask: u8,
}

pub struct AttentionInvokeF64<'a> {
    pub q: &'a [f64],
    pub k: &'a [f64],
    pub v: &'a [f64],
    pub out: &'a mut [f64],
    pub scratch_scores: &'a mut [f64],
    pub q_len: usize,
    pub k_len: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub k_stride: usize,
    pub v_stride: usize,
    pub head_offset: usize,
    pub mask: u8,
}

pub struct AdamwInvokeF32<'a> {
    pub params: &'a mut [f32],
    pub grads: &'a [f32],
    pub m: &'a mut [f32],
    pub v: &'a mut [f32],
    pub learning_rate: f32,
    pub step: u32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

pub struct AdamwInvokeF64<'a> {
    pub params: &'a mut [f64],
    pub grads: &'a [f64],
    pub m: &'a mut [f64],
    pub v: &'a mut [f64],
    pub learning_rate: f64,
    pub step: u32,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

pub(super) fn contract_abi_version_for_backend(backend: ComputeBackend) -> Option<u16> {
    match backend {
        ComputeBackend::Cpu => None,
        ComputeBackend::Gpu => Some(GPU_CONTRACT_ABI_VERSION),
        ComputeBackend::Tpu => Some(TPU_CONTRACT_ABI_VERSION),
        ComputeBackend::Lpu => Some(LPU_CONTRACT_ABI_VERSION),
    }
}

pub fn is_contract_abi_compatible(backend: ComputeBackend, abi_version: u16) -> bool {
    contract_abi_version_for_backend(backend)
        .map(|v| v == abi_version)
        .unwrap_or(false)
}

pub fn build_flop_abstraction_with_l_scale(
    flops_per_second: u128,
    lflop_scale_divisor: u128,
) -> FlopAbstraction {
    let l_scale = if lflop_scale_divisor == 0 {
        FLOPS_SCALE_LARGE
    } else {
        lflop_scale_divisor
    };
    FlopAbstraction {
        flops_per_second,
        gflops_x1000: flops_per_second.saturating_mul(1000) / FLOPS_SCALE_GIGA,
        tflops_x1000: flops_per_second.saturating_mul(1000) / FLOPS_SCALE_TERA,
        lflops_x1000: flops_per_second.saturating_mul(1000) / l_scale,
    }
}

pub fn build_flop_abstraction(flops_per_second: u128) -> FlopAbstraction {
    build_flop_abstraction_with_l_scale(flops_per_second, FLOPS_SCALE_LARGE)
}

pub fn build_ram_abstraction(total_bytes: u128, available_bytes: u128) -> RamAbstraction {
    let total = total_bytes;
    let available = available_bytes.min(total);
    let used = total.saturating_sub(available);
    let mib = 1024u128 * 1024u128;

    RamAbstraction {
        total_bytes: total,
        available_bytes: available,
        used_bytes: used,
        total_mib: total / mib,
        available_mib: available / mib,
        used_mib: used / mib,
    }
}

pub(super) const ALL_OPCODE_MASK: u32 = (1u32 << 18) - 1;

pub(super) const GPU_MATMUL_OPCODE_MASK: u32 =
    (1u32 << ((GpuOpCode::F32 as u32) - 1)) | (1u32 << ((GpuOpCode::F64 as u32) - 1));

pub(super) fn opcode_bit(opcode: GpuOpCode) -> u32 {
    1u32 << ((opcode as u32) - 1)
}

pub(super) fn opcode_from_raw(raw: u32) -> Option<GpuOpCode> {
    match raw {
        1 => Some(GpuOpCode::F32),
        2 => Some(GpuOpCode::F64),
        3 => Some(GpuOpCode::SoftmaxF32),
        4 => Some(GpuOpCode::SoftmaxF64),
        5 => Some(GpuOpCode::LayerNormF32),
        6 => Some(GpuOpCode::LayerNormF64),
        7 => Some(GpuOpCode::RmsNormF32),
        8 => Some(GpuOpCode::RmsNormF64),
        9 => Some(GpuOpCode::AttentionF32),
        10 => Some(GpuOpCode::AttentionF64),
        11 => Some(GpuOpCode::QuantizeI8F32),
        12 => Some(GpuOpCode::QuantizeI8F64),
        13 => Some(GpuOpCode::DequantizeI8F32),
        14 => Some(GpuOpCode::DequantizeI8F64),
        15 => Some(GpuOpCode::SgdF32),
        16 => Some(GpuOpCode::SgdF64),
        17 => Some(GpuOpCode::AdamwF32),
        18 => Some(GpuOpCode::AdamwF64),
        _ => None,
    }
}

pub(super) fn contract_reject_reason_from_raw(raw: u32) -> Option<ContractRejectReason> {
    match raw {
        1 => Some(ContractRejectReason::NoActiveBackendContract),
        2 => Some(ContractRejectReason::OpcodeNotSupported),
        3 => Some(ContractRejectReason::PayloadSizeMismatch),
        4 => Some(ContractRejectReason::NullPayloadPointer),
        5 => Some(ContractRejectReason::NoActiveContractInfo),
        6 => Some(ContractRejectReason::NoRegisteredHandler),
        7 => Some(ContractRejectReason::HandlerReturnedNonOk),
        8 => Some(ContractRejectReason::UnsupportedRequestFlags),
        9 => Some(ContractRejectReason::InsufficientSustainedFlops),
        10 => Some(ContractRejectReason::InsufficientAvailableMemory),
        _ => None,
    }
}

pub(super) fn backend_from_raw(raw: u8) -> Option<ComputeBackend> {
    match raw {
        0 => Some(ComputeBackend::Cpu),
        1 => Some(ComputeBackend::Gpu),
        2 => Some(ComputeBackend::Tpu),
        3 => Some(ComputeBackend::Lpu),
        _ => None,
    }
}

pub(super) fn expected_payload_len(opcode: GpuOpCode) -> usize {
    match opcode {
        GpuOpCode::F32 => core::mem::size_of::<KernelCmdF32>(),
        GpuOpCode::F64 => core::mem::size_of::<KernelCmdF64>(),
        GpuOpCode::SoftmaxF32 => core::mem::size_of::<SoftmaxCmdF32>(),
        GpuOpCode::SoftmaxF64 => core::mem::size_of::<SoftmaxCmdF64>(),
        GpuOpCode::LayerNormF32 => core::mem::size_of::<LayerNormCmdF32>(),
        GpuOpCode::LayerNormF64 => core::mem::size_of::<LayerNormCmdF64>(),
        GpuOpCode::RmsNormF32 => core::mem::size_of::<RmsNormCmdF32>(),
        GpuOpCode::RmsNormF64 => core::mem::size_of::<RmsNormCmdF64>(),
        GpuOpCode::AttentionF32 => core::mem::size_of::<AttentionCmdF32>(),
        GpuOpCode::AttentionF64 => core::mem::size_of::<AttentionCmdF64>(),
        GpuOpCode::QuantizeI8F32 => core::mem::size_of::<QuantizeI8CmdF32>(),
        GpuOpCode::QuantizeI8F64 => core::mem::size_of::<QuantizeI8CmdF64>(),
        GpuOpCode::DequantizeI8F32 => core::mem::size_of::<DequantizeI8CmdF32>(),
        GpuOpCode::DequantizeI8F64 => core::mem::size_of::<DequantizeI8CmdF64>(),
        GpuOpCode::SgdF32 => core::mem::size_of::<SgdCmdF32>(),
        GpuOpCode::SgdF64 => core::mem::size_of::<SgdCmdF64>(),
        GpuOpCode::AdamwF32 => core::mem::size_of::<AdamwCmdF32>(),
        GpuOpCode::AdamwF64 => core::mem::size_of::<AdamwCmdF64>(),
    }
}

pub(super) fn feature_flags_from_opcode_mask(opcode_mask: u32) -> u32 {
    let has_f32 = (opcode_mask
        & (opcode_bit(GpuOpCode::F32)
            | opcode_bit(GpuOpCode::SoftmaxF32)
            | opcode_bit(GpuOpCode::LayerNormF32)
            | opcode_bit(GpuOpCode::RmsNormF32)
            | opcode_bit(GpuOpCode::AttentionF32)
            | opcode_bit(GpuOpCode::SgdF32)
            | opcode_bit(GpuOpCode::AdamwF32)))
        != 0;
    let has_f64 = (opcode_mask
        & (opcode_bit(GpuOpCode::F64)
            | opcode_bit(GpuOpCode::SoftmaxF64)
            | opcode_bit(GpuOpCode::LayerNormF64)
            | opcode_bit(GpuOpCode::RmsNormF64)
            | opcode_bit(GpuOpCode::AttentionF64)
            | opcode_bit(GpuOpCode::SgdF64)
            | opcode_bit(GpuOpCode::AdamwF64)))
        != 0;
    let has_int8 = (opcode_mask
        & (opcode_bit(GpuOpCode::QuantizeI8F32)
            | opcode_bit(GpuOpCode::QuantizeI8F64)
            | opcode_bit(GpuOpCode::DequantizeI8F32)
            | opcode_bit(GpuOpCode::DequantizeI8F64)))
        != 0;

    let mut flags = 0u32;
    if has_f32 {
        flags |= CONTRACT_FEATURE_F32;
    }
    if has_f64 {
        flags |= CONTRACT_FEATURE_F64;
    }
    if has_int8 {
        flags |= CONTRACT_FEATURE_INT8;
    }
    flags | CONTRACT_FEATURE_BATCH
}

pub fn host_os_flags_from_name(os_name: &str) -> u32 {
    if os_name.eq_ignore_ascii_case("linux") {
        HOST_OS_FLAG_LINUX
    } else if os_name.eq_ignore_ascii_case("windows") {
        HOST_OS_FLAG_WINDOWS
    } else if os_name.eq_ignore_ascii_case("macos")
        || os_name.eq_ignore_ascii_case("darwin")
        || os_name.eq_ignore_ascii_case("mac")
    {
        HOST_OS_FLAG_MACOS
    } else {
        HOST_OS_FLAG_UNKNOWN
    }
}
