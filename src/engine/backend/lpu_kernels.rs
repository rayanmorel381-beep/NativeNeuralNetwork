use super::contract::register_lpu_command_handler_with_hardware_profile;
use super::types::{
    ALL_OPCODE_MASK, DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND, DEFAULT_LPU_TOTAL_MEMORY_BYTES,
    GpuContractRequest, GpuDispatchStatus, GpuOpCode, KernelCmdF32, KernelCmdF64,
    LayerNormCmdF32, LayerNormCmdF64, RmsNormCmdF32, RmsNormCmdF64, SoftmaxCmdF32,
    SoftmaxCmdF64, LPU_CONTRACT_ABI_VERSION, LPU_CONTRACT_MAGIC,
    AttentionCmdF32, AttentionCmdF64, QuantizeI8CmdF32, QuantizeI8CmdF64,
    DequantizeI8CmdF32, DequantizeI8CmdF64, SgdCmdF32, SgdCmdF64, AdamwCmdF32, AdamwCmdF64,
};
use crate::engine::runtime::hardware::arm::contains_ascii_nocase;
use crate::engine::runtime::hardware::arm::macos::{
    e_core_count, gpu, host_page_size, logical_cores, lpu, p_core_count, sysctl_string,
    sysctl_u64, unified_memory_bytes,
};
use core::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};

static LPU_INSTALLED: AtomicU8 = AtomicU8::new(0);
static LPU_DISPATCH_TOTAL: AtomicUsize = AtomicUsize::new(0);
static LPU_LAST_USER_CTX: AtomicUsize = AtomicUsize::new(0);
static LPU_LAST_OPCODE: AtomicU32 = AtomicU32::new(0);
static LPU_LAST_STATUS: AtomicU32 = AtomicU32::new(0);

const TOKEN_PREFILL: u32 = 0x01;
const TOKEN_DECODE: u32 = 0x02;
const TOKEN_SPECULATIVE: u32 = 0x04;

const HEALTH_CHECK_INTERVAL: usize = 1024;

fn opcode_token_kind(opcode: u32) -> u32 {
    match opcode {
        x if x == GpuOpCode::AttentionF32 as u32 || x == GpuOpCode::AttentionF64 as u32 => {
            TOKEN_DECODE
        }
        x if x == GpuOpCode::F32 as u32 || x == GpuOpCode::F64 as u32 => TOKEN_PREFILL,
        _ => TOKEN_SPECULATIVE,
    }
}

fn expected_payload_len(opcode: u32) -> usize {
    match opcode {
        x if x == GpuOpCode::F32 as u32 => core::mem::size_of::<KernelCmdF32>(),
        x if x == GpuOpCode::F64 as u32 => core::mem::size_of::<KernelCmdF64>(),
        x if x == GpuOpCode::SoftmaxF32 as u32 => core::mem::size_of::<SoftmaxCmdF32>(),
        x if x == GpuOpCode::SoftmaxF64 as u32 => core::mem::size_of::<SoftmaxCmdF64>(),
        x if x == GpuOpCode::LayerNormF32 as u32 => core::mem::size_of::<LayerNormCmdF32>(),
        x if x == GpuOpCode::LayerNormF64 as u32 => core::mem::size_of::<LayerNormCmdF64>(),
        x if x == GpuOpCode::RmsNormF32 as u32 => core::mem::size_of::<RmsNormCmdF32>(),
        x if x == GpuOpCode::RmsNormF64 as u32 => core::mem::size_of::<RmsNormCmdF64>(),
        x if x == GpuOpCode::AttentionF32 as u32 => core::mem::size_of::<AttentionCmdF32>(),
        x if x == GpuOpCode::AttentionF64 as u32 => core::mem::size_of::<AttentionCmdF64>(),
        x if x == GpuOpCode::QuantizeI8F32 as u32 => core::mem::size_of::<QuantizeI8CmdF32>(),
        x if x == GpuOpCode::QuantizeI8F64 as u32 => core::mem::size_of::<QuantizeI8CmdF64>(),
        x if x == GpuOpCode::DequantizeI8F32 as u32 => core::mem::size_of::<DequantizeI8CmdF32>(),
        x if x == GpuOpCode::DequantizeI8F64 as u32 => core::mem::size_of::<DequantizeI8CmdF64>(),
        x if x == GpuOpCode::SgdF32 as u32 => core::mem::size_of::<SgdCmdF32>(),
        x if x == GpuOpCode::SgdF64 as u32 => core::mem::size_of::<SgdCmdF64>(),
        x if x == GpuOpCode::AdamwF32 as u32 => core::mem::size_of::<AdamwCmdF32>(),
        x if x == GpuOpCode::AdamwF64 as u32 => core::mem::size_of::<AdamwCmdF64>(),
        _ => 0,
    }
}

fn token_count(opcode: u32, payload_ptr: *const u8) -> u32 {
    if payload_ptr.is_null() {
        return 1;
    }
    let n = match opcode {
        x if x == GpuOpCode::AttentionF32 as u32 => unsafe {
            (*(payload_ptr as *const AttentionCmdF32)).q_len
        },
        x if x == GpuOpCode::AttentionF64 as u32 => unsafe {
            (*(payload_ptr as *const AttentionCmdF64)).q_len
        },
        x if x == GpuOpCode::F32 as u32 => unsafe {
            (*(payload_ptr as *const KernelCmdF32)).batch_size
        },
        x if x == GpuOpCode::F64 as u32 => unsafe {
            (*(payload_ptr as *const KernelCmdF64)).batch_size
        },
        x if x == GpuOpCode::SoftmaxF32 as u32 => unsafe {
            (*(payload_ptr as *const SoftmaxCmdF32)).len
        },
        x if x == GpuOpCode::SoftmaxF64 as u32 => unsafe {
            (*(payload_ptr as *const SoftmaxCmdF64)).len
        },
        x if x == GpuOpCode::LayerNormF32 as u32 => unsafe {
            (*(payload_ptr as *const LayerNormCmdF32)).len
        },
        x if x == GpuOpCode::LayerNormF64 as u32 => unsafe {
            (*(payload_ptr as *const LayerNormCmdF64)).len
        },
        x if x == GpuOpCode::RmsNormF32 as u32 => unsafe {
            (*(payload_ptr as *const RmsNormCmdF32)).len
        },
        x if x == GpuOpCode::RmsNormF64 as u32 => unsafe {
            (*(payload_ptr as *const RmsNormCmdF64)).len
        },
        x if x == GpuOpCode::QuantizeI8F32 as u32 => unsafe {
            (*(payload_ptr as *const QuantizeI8CmdF32)).input_len
        },
        x if x == GpuOpCode::QuantizeI8F64 as u32 => unsafe {
            (*(payload_ptr as *const QuantizeI8CmdF64)).input_len
        },
        x if x == GpuOpCode::DequantizeI8F32 as u32 => unsafe {
            (*(payload_ptr as *const DequantizeI8CmdF32)).input_len
        },
        x if x == GpuOpCode::DequantizeI8F64 as u32 => unsafe {
            (*(payload_ptr as *const DequantizeI8CmdF64)).input_len
        },
        x if x == GpuOpCode::SgdF32 as u32 => unsafe {
            (*(payload_ptr as *const SgdCmdF32)).len
        },
        x if x == GpuOpCode::SgdF64 as u32 => unsafe {
            (*(payload_ptr as *const SgdCmdF64)).len
        },
        x if x == GpuOpCode::AdamwF32 as u32 => unsafe {
            (*(payload_ptr as *const AdamwCmdF32)).len
        },
        x if x == GpuOpCode::AdamwF64 as u32 => unsafe {
            (*(payload_ptr as *const AdamwCmdF64)).len
        },
        _ => 1,
    };
    n.min(u32::MAX as usize).max(1) as u32
}

fn element_size(opcode: u32) -> usize {
    match opcode {
        x if x == GpuOpCode::F64 as u32
            || x == GpuOpCode::SoftmaxF64 as u32
            || x == GpuOpCode::LayerNormF64 as u32
            || x == GpuOpCode::RmsNormF64 as u32
            || x == GpuOpCode::AttentionF64 as u32
            || x == GpuOpCode::QuantizeI8F64 as u32
            || x == GpuOpCode::DequantizeI8F64 as u32
            || x == GpuOpCode::SgdF64 as u32
            || x == GpuOpCode::AdamwF64 as u32 => 8,
        x if x == GpuOpCode::QuantizeI8F32 as u32 || x == GpuOpCode::DequantizeI8F32 as u32 => 1,
        _ => 4,
    }
}

fn lpu_command_handler(request: *const GpuContractRequest, user_ctx: usize) -> u32 {
    if request.is_null() {
        return GpuDispatchStatus::BadContract as u32;
    }
    LPU_LAST_USER_CTX.store(user_ctx, Ordering::Relaxed);
    let count = LPU_DISPATCH_TOTAL.fetch_add(1, Ordering::AcqRel) + 1;

    let header = unsafe { (*request).header };
    if header.magic != LPU_CONTRACT_MAGIC || header.abi_version != LPU_CONTRACT_ABI_VERSION {
        return GpuDispatchStatus::BadContract as u32;
    }
    LPU_LAST_OPCODE.store(header.opcode, Ordering::Relaxed);

    let payload_ptr = unsafe { (*request).payload_ptr };
    if payload_ptr.is_null() {
        return GpuDispatchStatus::BadPayload as u32;
    }
    let expected = expected_payload_len(header.opcode);
    if expected == 0 {
        return GpuDispatchStatus::NotSupported as u32;
    }
    if header.payload_len as usize != expected {
        return GpuDispatchStatus::BadPayload as u32;
    }

    let cfg = lpu::default_config();
    let workers = lpu::clamp_workers(cfg.compute_queues).max(1);
    let tokens = token_count(header.opcode, payload_ptr);
    let sched = lpu::build_schedule(tokens as usize * workers);
    if sched.chunks == 0 || sched.chunk_size == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    if !lpu::device_is_initialized() && !lpu::probe_device() {
        return GpuDispatchStatus::Failed as u32;
    }

    let base = lpu::device_mmio_base();
    if base == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    if count.is_multiple_of(HEALTH_CHECK_INTERVAL) && lpu::run_diagnostics() == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    if !lpu::is_powered() {
        lpu::power_cycle();
    }
    if !lpu::dma_region_ready() {
        return GpuDispatchStatus::Failed as u32;
    }
    if lpu::dma_remaining() == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    let elem = element_size(header.opcode);
    let kind = opcode_token_kind(header.opcode);
    let dev_id = lpu::device_id();
    let revision = lpu::revision();
    LPU_LAST_STATUS.store(dev_id ^ revision, Ordering::Relaxed);

    let mut completed = 0usize;
    let mut iter = 0usize;
    while iter < sched.chunks {
        let off = iter * sched.chunk_size;
        if off as u32 >= tokens {
            break;
        }
        let chunk_tokens = ((tokens as usize - off).min(sched.chunk_size)).max(1);
        let bytes = chunk_tokens.saturating_mul(elem).max(elem);
        let buf = lpu::dma_alloc(bytes);
        if buf == 0 {
            return GpuDispatchStatus::Failed as u32;
        }
        if !lpu::dma_submit(buf as u32, buf as u32, bytes as u32, true) {
            return GpuDispatchStatus::Failed as u32;
        }
        let mut poll = 0u32;
        while poll < 1024 && !lpu::dma_complete() {
            poll = poll.wrapping_add(1);
        }
        if lpu::dma_error() {
            return GpuDispatchStatus::Failed as u32;
        }
        let dispatched = match kind {
            TOKEN_PREFILL => lpu::submit_prefill(buf as u32, chunk_tokens as u32),
            TOKEN_DECODE => lpu::submit_decode(buf as u32),
            _ => lpu::submit_speculative(buf as u32, chunk_tokens as u32),
        };
        if !dispatched {
            return GpuDispatchStatus::Failed as u32;
        }
        let status = lpu::read_status_reg();
        LPU_LAST_STATUS.store(status, Ordering::Relaxed);
        if status & 0x10 != 0 {
            return GpuDispatchStatus::Failed as u32;
        }
        let scratch = lpu::read_reg(0);
        lpu::write_reg(4, scratch.wrapping_add(kind));
        completed = completed.saturating_add(chunk_tokens);
        iter += 1;
    }

    if completed == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    GpuDispatchStatus::Ok as u32
}

pub(crate) fn install() {
    if LPU_INSTALLED.swap(1, Ordering::AcqRel) != 0 {
        return;
    }

    let logical = logical_cores().max(1);
    let page = host_page_size().max(4096);
    let p_cores = p_core_count().max(1);
    let e_cores = e_core_count();
    let mem = unified_memory_bytes();

    let gpu_cfg = gpu::default_config();
    let gpu_clamp = gpu::clamp_workers(gpu_cfg.gpu_cores.max(1));
    let gpu_sched = gpu::build_schedule(gpu_cfg.gpu_cores.max(1));

    let mut brand_buf = [0u8; 96];
    let brand_len =
        sysctl_string(b"machdep.cpu.brand_string\0", &mut brand_buf).unwrap_or(0);
    let brand_str = core::str::from_utf8(&brand_buf[..brand_len]).unwrap_or("");
    let is_apple = contains_ascii_nocase(brand_str, "apple");
    let hint = sysctl_u64(b"hw.memsize\0").unwrap_or(0);

    lpu::configure_device(0, 0, p_cores as u32);
    let probed = lpu::probe_device();
    let cfg = lpu::default_config();
    let clamp = lpu::clamp_workers(cfg.compute_queues).max(1);
    let sched = lpu::build_schedule(cfg.workgroup_size.max(1));
    let diag = lpu::run_diagnostics();

    let init_flag = lpu::device_is_initialized() as usize;
    let base = lpu::device_mmio_base();
    let status = lpu::read_status_reg();
    let reg = lpu::read_reg(0);
    lpu::write_reg(0, reg);
    let pref_ok = lpu::submit_prefill(0, 0) as usize;
    let dec_ok = lpu::submit_decode(0) as usize;
    let spec_ok = lpu::submit_speculative(0, 0) as usize;
    lpu::power_cycle();
    let powered = lpu::is_powered() as usize;
    lpu::install_irq(p_cores as u32, 0);
    let rev = lpu::revision();
    let dev_id = lpu::device_id();
    let dma_buf = lpu::dma_alloc(page);
    let dma_done = lpu::dma_submit(dma_buf as u32, dma_buf as u32, page as u32, true) as usize;
    let dma_ready = lpu::dma_complete() as usize;
    let dma_err = lpu::dma_error() as usize;
    let region_ready = lpu::dma_region_ready() as usize;
    let remaining = lpu::dma_remaining();
    let streams = lpu::smmu_streams();
    let mapped = lpu::smmu_map(base, page);
    lpu::smmu_set_attrs(p_cores as u32, p_cores as u32);
    let mapped_bytes = lpu::smmu_mapped_bytes();

    let mem_total = if hint > 0 {
        hint
    } else {
        mem.max(DEFAULT_LPU_TOTAL_MEMORY_BYTES)
    };
    let mem_avail = mem_total - (mem_total >> 4);
    let flops = if is_apple {
        DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND
    } else {
        DEFAULT_LPU_SUSTAINED_FLOPS_PER_SECOND / 2
    };

    let seed = (diag ^ init_flag ^ base ^ mapped ^ mapped_bytes ^ dma_buf ^ remaining)
        .wrapping_add(status as usize)
        .wrapping_add(reg as usize)
        .wrapping_add(pref_ok)
        .wrapping_add(dec_ok)
        .wrapping_add(spec_ok)
        .wrapping_add(powered)
        .wrapping_add(rev as usize)
        .wrapping_add(dev_id as usize)
        .wrapping_add(dma_done)
        .wrapping_add(dma_ready)
        .wrapping_add(dma_err)
        .wrapping_add(region_ready)
        .wrapping_add(streams as usize)
        .wrapping_add(logical)
        .wrapping_add(page)
        .wrapping_add(p_cores)
        .wrapping_add(e_cores)
        .wrapping_add(clamp)
        .wrapping_add(sched.chunks)
        .wrapping_add(sched.chunk_size)
        .wrapping_add(sched.frame_budget_us as usize)
        .wrapping_add(gpu_clamp)
        .wrapping_add(gpu_sched.chunks)
        .wrapping_add(gpu_sched.chunk_size)
        .wrapping_add(gpu_sched.frame_budget_us as usize)
        .wrapping_add(gpu_cfg.gpu_cores)
        .wrapping_add(gpu_cfg.metal_queues)
        .wrapping_add(gpu_cfg.tile_size)
        .wrapping_add(gpu_cfg.simd_width)
        .wrapping_add(gpu_cfg.tile_memory_bytes)
        .wrapping_add(gpu_cfg.unified_memory_bytes as usize)
        .wrapping_add(gpu_cfg.frame_budget_us as usize)
        .wrapping_add(gpu_cfg.low_power as usize)
        .wrapping_add(cfg.render_threads)
        .wrapping_add(cfg.double_buffered as usize)
        .wrapping_add(cfg.frame_budget_us as usize)
        .wrapping_add(cfg.low_power as usize)
        .wrapping_add(probed as usize);

    register_lpu_command_handler_with_hardware_profile(
        lpu_command_handler,
        seed,
        ALL_OPCODE_MASK,
        flops as u128,
        mem_total as u128,
        mem_avail as u128,
    );
}

pub fn lpu_dispatch_count() -> usize {
    LPU_DISPATCH_TOTAL.load(Ordering::Relaxed)
}

pub fn lpu_last_status() -> u32 {
    LPU_LAST_STATUS.load(Ordering::Relaxed)
}

pub fn lpu_last_opcode() -> u32 {
    LPU_LAST_OPCODE.load(Ordering::Relaxed)
}
