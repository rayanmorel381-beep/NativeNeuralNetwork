use super::contract::{
    clear_gpu_command_handler, clear_gpu_kernel_f32, clear_gpu_kernel_f64,
    clear_last_contract_reject, register_contract_reject_handler,
    register_gpu_command_handler_with_hardware_profile, register_gpu_kernel_f32,
    register_gpu_kernel_f64, set_hardware_contract_strict, validate_contract_request_for_backend,
};
use super::types::{
    ComputeBackend, ContractRejectReason, DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND,
    DEFAULT_GPU_TOTAL_MEMORY_BYTES, GPU_MATMUL_OPCODE_MASK, GpuContractRequest, GpuDispatchStatus,
    GpuOpCode, KernelCmdF32, KernelCmdF64, KernelInvokeF32, KernelInvokeF64,
};
use super::contract::set_compute_backend;
use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

static GPU_READY: AtomicU8 = AtomicU8::new(0);
static GPU_BAR0: AtomicUsize = AtomicUsize::new(0);
static GPU_VENDOR: AtomicUsize = AtomicUsize::new(0);
static GPU_DEVICE: AtomicUsize = AtomicUsize::new(0);
static GPU_DISPATCH_TOTAL: AtomicUsize = AtomicUsize::new(0);
static GPU_LAST_USER_CTX: AtomicUsize = AtomicUsize::new(0);

#[allow(clippy::too_many_arguments)]
fn stage_and_dispatch(
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
    if GPU_BAR0.load(Ordering::Acquire) == 0 {
        return false;
    }
    GPU_DISPATCH_TOTAL.fetch_add(1, Ordering::AcqRel);
    crate::engine::runtime::gpu_matmul(
        src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
    )
}

fn stage_and_dispatch_f32(args: KernelInvokeF32<'_>) -> bool {
    let src = unsafe { core::slice::from_raw_parts(args.src.as_ptr() as *const u8, args.src.len() * 4) };
    let weights = unsafe { core::slice::from_raw_parts(args.weights.as_ptr() as *const u8, args.weights.len() * 4) };
    let biases = unsafe { core::slice::from_raw_parts(args.biases.as_ptr() as *const u8, args.biases.len() * 4) };
    let dst = unsafe { core::slice::from_raw_parts_mut(args.dst.as_mut_ptr() as *mut u8, args.dst.len() * 4) };
    stage_and_dispatch(
        src, dst, args.batch_size, args.stride, args.in_size, args.out_size, weights, biases,
        args.activation.to_u8() as u32,
    )
}

fn stage_and_dispatch_f64(args: KernelInvokeF64<'_>) -> bool {
    let src = unsafe { core::slice::from_raw_parts(args.src.as_ptr() as *const u8, args.src.len() * 8) };
    let weights = unsafe { core::slice::from_raw_parts(args.weights.as_ptr() as *const u8, args.weights.len() * 8) };
    let biases = unsafe { core::slice::from_raw_parts(args.biases.as_ptr() as *const u8, args.biases.len() * 8) };
    let dst = unsafe { core::slice::from_raw_parts_mut(args.dst.as_mut_ptr() as *mut u8, args.dst.len() * 8) };
    stage_and_dispatch(
        src, dst, args.batch_size, args.stride, args.in_size, args.out_size, weights, biases,
        args.activation.to_u8() as u32,
    )
}

fn gpu_kernel_f32(args: KernelInvokeF32<'_>) -> bool {
    if GPU_READY.load(Ordering::Acquire) == 0 {
        return false;
    }
    if args.weights.len() < args.out_size * args.in_size || args.biases.len() < args.out_size {
        return false;
    }
    if args.dst.len() < args.batch_size * args.stride {
        return false;
    }
    stage_and_dispatch_f32(args)
}

fn gpu_kernel_f64(args: KernelInvokeF64<'_>) -> bool {
    if GPU_READY.load(Ordering::Acquire) == 0 {
        return false;
    }
    if args.weights.len() < args.out_size * args.in_size || args.biases.len() < args.out_size {
        return false;
    }
    if args.dst.len() < args.batch_size * args.stride {
        return false;
    }
    stage_and_dispatch_f64(args)
}

fn gpu_command_handler(request: *const GpuContractRequest, user_ctx: usize) -> u32 {
    if request.is_null() {
        return GpuDispatchStatus::BadContract as u32;
    }
    GPU_LAST_USER_CTX.store(user_ctx, Ordering::Relaxed);
    GPU_DISPATCH_TOTAL.fetch_add(1, Ordering::AcqRel);
    let req = unsafe { &*request };
    let status = validate_contract_request_for_backend(ComputeBackend::Gpu, req);
    if status as u32 != GpuDispatchStatus::Ok as u32 {
        return status as u32;
    }
    if GPU_BAR0.load(Ordering::Acquire) == 0 {
        return GpuDispatchStatus::Failed as u32;
    }
    let header = req.header;
    let payload_len = header.payload_len as usize;
    let ok = match header.opcode {
        x if x == GpuOpCode::F32 as u32 => dispatch_matmul_cmd_f32(req.payload_ptr, payload_len),
        x if x == GpuOpCode::F64 as u32 => dispatch_matmul_cmd_f64(req.payload_ptr, payload_len),
        _ => return GpuDispatchStatus::NotSupported as u32,
    };
    if ok {
        GpuDispatchStatus::Ok as u32
    } else {
        GpuDispatchStatus::Failed as u32
    }
}

fn dispatch_matmul_cmd_f32(payload_ptr: *const u8, payload_len: usize) -> bool {
    if payload_ptr.is_null() || payload_len != core::mem::size_of::<KernelCmdF32>() {
        return false;
    }
    let cmd = unsafe { &*(payload_ptr as *const KernelCmdF32) };
    if cmd.src_ptr.is_null() || cmd.dst_ptr.is_null() || cmd.weights_ptr.is_null() {
        return false;
    }
    let src = unsafe { core::slice::from_raw_parts(cmd.src_ptr as *const u8, cmd.src_len * 4) };
    let dst = unsafe { core::slice::from_raw_parts_mut(cmd.dst_ptr as *mut u8, cmd.dst_len * 4) };
    let weights =
        unsafe { core::slice::from_raw_parts(cmd.weights_ptr as *const u8, cmd.weights_len * 4) };
    let biases = if cmd.biases_ptr.is_null() {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(cmd.biases_ptr as *const u8, cmd.biases_len * 4) }
    };
    stage_and_dispatch(
        src, dst, cmd.batch_size, cmd.stride, cmd.in_size, cmd.out_size, weights, biases,
        cmd.activation as u32,
    )
}

fn dispatch_matmul_cmd_f64(payload_ptr: *const u8, payload_len: usize) -> bool {
    if payload_ptr.is_null() || payload_len != core::mem::size_of::<KernelCmdF64>() {
        return false;
    }
    let cmd = unsafe { &*(payload_ptr as *const KernelCmdF64) };
    if cmd.src_ptr.is_null() || cmd.dst_ptr.is_null() || cmd.weights_ptr.is_null() {
        return false;
    }
    let src = unsafe { core::slice::from_raw_parts(cmd.src_ptr as *const u8, cmd.src_len * 8) };
    let dst = unsafe { core::slice::from_raw_parts_mut(cmd.dst_ptr as *mut u8, cmd.dst_len * 8) };
    let weights =
        unsafe { core::slice::from_raw_parts(cmd.weights_ptr as *const u8, cmd.weights_len * 8) };
    let biases = if cmd.biases_ptr.is_null() {
        &[][..]
    } else {
        unsafe { core::slice::from_raw_parts(cmd.biases_ptr as *const u8, cmd.biases_len * 8) }
    };
    stage_and_dispatch(
        src, dst, cmd.batch_size, cmd.stride, cmd.in_size, cmd.out_size, weights, biases,
        cmd.activation as u32,
    )
}

pub(crate) fn activate_gpu_backend(bar0: usize, vendor_id: usize, device_id: usize) {
    GPU_BAR0.store(bar0, Ordering::Release);
    GPU_VENDOR.store(vendor_id, Ordering::Release);
    GPU_DEVICE.store(device_id, Ordering::Release);
    register_gpu_kernel_f32(gpu_kernel_f32);
    register_gpu_kernel_f64(gpu_kernel_f64);
    set_compute_backend(ComputeBackend::Gpu);
    GPU_READY.store(1, Ordering::Release);
}

fn gpu_reject_circuit_breaker(
    reason: ContractRejectReason,
    backend: ComputeBackend,
    _opcode: u32,
    _user_ctx: usize,
) {
    if backend != ComputeBackend::Gpu {
        return;
    }
    let fatal = matches!(
        reason,
        ContractRejectReason::NoActiveBackendContract
            | ContractRejectReason::NoActiveContractInfo
            | ContractRejectReason::NoRegisteredHandler
            | ContractRejectReason::InsufficientSustainedFlops
            | ContractRejectReason::InsufficientAvailableMemory
    );
    if !fatal {
        return;
    }
    GPU_READY.store(0, Ordering::Release);
    GPU_BAR0.store(0, Ordering::Release);
    clear_gpu_kernel_f32();
    clear_gpu_kernel_f64();
    clear_gpu_command_handler();
    clear_last_contract_reject();
}

pub(crate) fn install() {
    match crate::engine::runtime::gpu_device_probe() {
        Some((device_id, _, _)) => activate_gpu_backend(1, 0, device_id as usize),
        None => return,
    }
    let bar0 = GPU_BAR0.load(Ordering::Acquire);
    let vendor = GPU_VENDOR.load(Ordering::Acquire);
    let device = GPU_DEVICE.load(Ordering::Acquire);
    let seed = bar0
        .wrapping_add(vendor)
        .wrapping_add(device)
        .wrapping_add(gpu_dispatch_count())
        .wrapping_add(GPU_LAST_USER_CTX.load(Ordering::Acquire));
    register_gpu_command_handler_with_hardware_profile(
        gpu_command_handler,
        seed,
        GPU_MATMUL_OPCODE_MASK,
        DEFAULT_GPU_SUSTAINED_FLOPS_PER_SECOND as u128,
        DEFAULT_GPU_TOTAL_MEMORY_BYTES as u128,
        (DEFAULT_GPU_TOTAL_MEMORY_BYTES - DEFAULT_GPU_TOTAL_MEMORY_BYTES / 8) as u128,
    );
    set_hardware_contract_strict(true);
    register_contract_reject_handler(gpu_reject_circuit_breaker, seed);
}

pub fn gpu_dispatch_count() -> usize {
    GPU_DISPATCH_TOTAL.load(Ordering::Relaxed)
}
