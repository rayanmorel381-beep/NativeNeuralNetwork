use super::contract::register_tpu_command_handler_with_hardware_profile;
use super::types::{
    ALL_OPCODE_MASK, DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND, DEFAULT_TPU_TOTAL_MEMORY_BYTES,
    GpuContractRequest, GpuDispatchStatus, GpuOpCode, KernelCmdF32, KernelCmdF64,
    LayerNormCmdF32, LayerNormCmdF64, RmsNormCmdF32, RmsNormCmdF64, SoftmaxCmdF32,
    SoftmaxCmdF64, TPU_CONTRACT_ABI_VERSION, TPU_CONTRACT_MAGIC,
    AttentionCmdF32, AttentionCmdF64, QuantizeI8CmdF32, QuantizeI8CmdF64,
    DequantizeI8CmdF32, DequantizeI8CmdF64, SgdCmdF32, SgdCmdF64, AdamwCmdF32, AdamwCmdF64,
};
use crate::engine::runtime::hardware::arm::contains_ascii_nocase;
use crate::engine::runtime::hardware::arm::macos::{
    cpu, cpu_family, host_page_size, logical_cores, ram, sysctl_string, sysctl_u32,
    sysctl_u64, tpu, unified_memory_bytes,
};
use core::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};

static TPU_INSTALLED: AtomicU8 = AtomicU8::new(0);
static TPU_DISPATCH_TOTAL: AtomicUsize = AtomicUsize::new(0);
static TPU_LAST_USER_CTX: AtomicUsize = AtomicUsize::new(0);
static TPU_LAST_STATUS: AtomicU32 = AtomicU32::new(0);
static TPU_LAST_OPCODE: AtomicU32 = AtomicU32::new(0);

const CMD_BASE_F32: u32 = 0x10;
const CMD_BASE_F64: u32 = 0x18;
const CMD_SOFTMAX_F32: u32 = 0x20;
const CMD_SOFTMAX_F64: u32 = 0x28;
const CMD_LAYER_NORM_F32: u32 = 0x30;
const CMD_LAYER_NORM_F64: u32 = 0x38;
const CMD_RMS_NORM_F32: u32 = 0x40;
const CMD_RMS_NORM_F64: u32 = 0x48;
const CMD_ATTENTION_F32: u32 = 0x50;
const CMD_ATTENTION_F64: u32 = 0x58;
const CMD_QUANT_F32: u32 = 0x60;
const CMD_QUANT_F64: u32 = 0x68;
const CMD_DEQUANT_F32: u32 = 0x70;
const CMD_DEQUANT_F64: u32 = 0x78;
const CMD_SGD_F32: u32 = 0x80;
const CMD_SGD_F64: u32 = 0x88;
const CMD_ADAMW_F32: u32 = 0x90;
const CMD_ADAMW_F64: u32 = 0x98;

const HEALTH_CHECK_INTERVAL: usize = 1024;

fn opcode_to_mmio_cmd(opcode: u32) -> Option<u32> {
    match opcode {
        x if x == GpuOpCode::F32 as u32 => Some(CMD_BASE_F32),
        x if x == GpuOpCode::F64 as u32 => Some(CMD_BASE_F64),
        x if x == GpuOpCode::SoftmaxF32 as u32 => Some(CMD_SOFTMAX_F32),
        x if x == GpuOpCode::SoftmaxF64 as u32 => Some(CMD_SOFTMAX_F64),
        x if x == GpuOpCode::LayerNormF32 as u32 => Some(CMD_LAYER_NORM_F32),
        x if x == GpuOpCode::LayerNormF64 as u32 => Some(CMD_LAYER_NORM_F64),
        x if x == GpuOpCode::RmsNormF32 as u32 => Some(CMD_RMS_NORM_F32),
        x if x == GpuOpCode::RmsNormF64 as u32 => Some(CMD_RMS_NORM_F64),
        x if x == GpuOpCode::AttentionF32 as u32 => Some(CMD_ATTENTION_F32),
        x if x == GpuOpCode::AttentionF64 as u32 => Some(CMD_ATTENTION_F64),
        x if x == GpuOpCode::QuantizeI8F32 as u32 => Some(CMD_QUANT_F32),
        x if x == GpuOpCode::QuantizeI8F64 as u32 => Some(CMD_QUANT_F64),
        x if x == GpuOpCode::DequantizeI8F32 as u32 => Some(CMD_DEQUANT_F32),
        x if x == GpuOpCode::DequantizeI8F64 as u32 => Some(CMD_DEQUANT_F64),
        x if x == GpuOpCode::SgdF32 as u32 => Some(CMD_SGD_F32),
        x if x == GpuOpCode::SgdF64 as u32 => Some(CMD_SGD_F64),
        x if x == GpuOpCode::AdamwF32 as u32 => Some(CMD_ADAMW_F32),
        x if x == GpuOpCode::AdamwF64 as u32 => Some(CMD_ADAMW_F64),
        _ => None,
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

fn workload_items(opcode: u32, payload_ptr: *const u8) -> usize {
    if payload_ptr.is_null() {
        return 0;
    }
    match opcode {
        x if x == GpuOpCode::F32 as u32 => unsafe {
            let cmd = payload_ptr as *const KernelCmdF32;
            (*cmd).batch_size.saturating_mul((*cmd).out_size).max(1)
        },
        x if x == GpuOpCode::F64 as u32 => unsafe {
            let cmd = payload_ptr as *const KernelCmdF64;
            (*cmd).batch_size.saturating_mul((*cmd).out_size).max(1)
        },
        x if x == GpuOpCode::SoftmaxF32 as u32 => unsafe {
            (*(payload_ptr as *const SoftmaxCmdF32)).len.max(1)
        },
        x if x == GpuOpCode::SoftmaxF64 as u32 => unsafe {
            (*(payload_ptr as *const SoftmaxCmdF64)).len.max(1)
        },
        x if x == GpuOpCode::LayerNormF32 as u32 => unsafe {
            (*(payload_ptr as *const LayerNormCmdF32)).len.max(1)
        },
        x if x == GpuOpCode::LayerNormF64 as u32 => unsafe {
            (*(payload_ptr as *const LayerNormCmdF64)).len.max(1)
        },
        x if x == GpuOpCode::RmsNormF32 as u32 => unsafe {
            (*(payload_ptr as *const RmsNormCmdF32)).len.max(1)
        },
        x if x == GpuOpCode::RmsNormF64 as u32 => unsafe {
            (*(payload_ptr as *const RmsNormCmdF64)).len.max(1)
        },
        x if x == GpuOpCode::AttentionF32 as u32 => unsafe {
            let cmd = payload_ptr as *const AttentionCmdF32;
            (*cmd).q_len.saturating_mul((*cmd).k_len).max(1)
        },
        x if x == GpuOpCode::AttentionF64 as u32 => unsafe {
            let cmd = payload_ptr as *const AttentionCmdF64;
            (*cmd).q_len.saturating_mul((*cmd).k_len).max(1)
        },
        x if x == GpuOpCode::QuantizeI8F32 as u32 => unsafe {
            (*(payload_ptr as *const QuantizeI8CmdF32)).input_len.max(1)
        },
        x if x == GpuOpCode::QuantizeI8F64 as u32 => unsafe {
            (*(payload_ptr as *const QuantizeI8CmdF64)).input_len.max(1)
        },
        x if x == GpuOpCode::DequantizeI8F32 as u32 => unsafe {
            (*(payload_ptr as *const DequantizeI8CmdF32)).input_len.max(1)
        },
        x if x == GpuOpCode::DequantizeI8F64 as u32 => unsafe {
            (*(payload_ptr as *const DequantizeI8CmdF64)).input_len.max(1)
        },
        x if x == GpuOpCode::SgdF32 as u32 => unsafe {
            (*(payload_ptr as *const SgdCmdF32)).len.max(1)
        },
        x if x == GpuOpCode::SgdF64 as u32 => unsafe {
            (*(payload_ptr as *const SgdCmdF64)).len.max(1)
        },
        x if x == GpuOpCode::AdamwF32 as u32 => unsafe {
            (*(payload_ptr as *const AdamwCmdF32)).len.max(1)
        },
        x if x == GpuOpCode::AdamwF64 as u32 => unsafe {
            (*(payload_ptr as *const AdamwCmdF64)).len.max(1)
        },
        _ => 1,
    }
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

fn tpu_command_handler(request: *const GpuContractRequest, user_ctx: usize) -> u32 {
    if request.is_null() {
        return GpuDispatchStatus::BadContract as u32;
    }
    TPU_LAST_USER_CTX.store(user_ctx, Ordering::Relaxed);
    let count = TPU_DISPATCH_TOTAL.fetch_add(1, Ordering::AcqRel) + 1;

    let header = unsafe { (*request).header };
    if header.magic != TPU_CONTRACT_MAGIC || header.abi_version != TPU_CONTRACT_ABI_VERSION {
        return GpuDispatchStatus::BadContract as u32;
    }
    TPU_LAST_OPCODE.store(header.opcode, Ordering::Relaxed);

    let cmd = match opcode_to_mmio_cmd(header.opcode) {
        Some(c) => c,
        None => return GpuDispatchStatus::NotSupported as u32,
    };
    let payload_ptr = unsafe { (*request).payload_ptr };
    if payload_ptr.is_null() {
        return GpuDispatchStatus::BadPayload as u32;
    }
    if header.payload_len as usize != expected_payload_len(header.opcode) {
        return GpuDispatchStatus::BadPayload as u32;
    }

    let cfg = tpu::default_config();
    let workers = tpu::clamp_workers(cfg.compute_queues).max(1);
    let items = workload_items(header.opcode, payload_ptr).max(workers);
    let sched = tpu::build_schedule(items);
    if sched.chunks == 0 || sched.chunk_size == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    if !tpu::device_is_initialized() && !tpu::probe_device() {
        return GpuDispatchStatus::Failed as u32;
    }

    let base = tpu::device_mmio_base();
    if base == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    if count.is_multiple_of(HEALTH_CHECK_INTERVAL) && tpu::run_diagnostics() == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    let elem = element_size(header.opcode);
    let region_base = tpu::dma_region_base();
    if region_base == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    let mut completed = 0usize;
    let mut iter = 0usize;
    while iter < sched.chunks {
        let off = iter * sched.chunk_size;
        if off >= items {
            break;
        }
        let len = (items - off).min(sched.chunk_size).max(1);
        let bytes = len.saturating_mul(elem).max(elem);
        let dma_addr = match tpu::dma_alloc(bytes) {
            Some(a) => a,
            None => return GpuDispatchStatus::Failed as u32,
        };
        tpu::dma_clean_invalidate(dma_addr, bytes);
        if !tpu::dma_submit(dma_addr as u64, bytes as u32) {
            return GpuDispatchStatus::Failed as u32;
        }
        let mut poll = 0u32;
        while poll < 1024 && !tpu::dma_complete() {
            poll = poll.wrapping_add(1);
        }
        let status = tpu::submit_compute(cmd, dma_addr as u32, bytes as u32);
        TPU_LAST_STATUS.store(status, Ordering::Relaxed);
        let live_status = tpu::read_status_reg();
        if live_status & 0x10 != 0 {
            return GpuDispatchStatus::Failed as u32;
        }
        let scratch = tpu::read_reg(0);
        tpu::write_reg(4, scratch.wrapping_add(cmd));
        completed = completed.saturating_add(len);
        iter += 1;
    }

    if completed == 0 {
        return GpuDispatchStatus::Failed as u32;
    }

    GpuDispatchStatus::Ok as u32
}

pub(crate) fn install() {
    if TPU_INSTALLED.swap(1, Ordering::AcqRel) != 0 {
        return;
    }

    let family = cpu_family();
    let logical = logical_cores().max(1);
    let page = host_page_size().max(4096);
    let cpu_info = cpu::default_config();
    let cpu_clamp = cpu::clamp_workers(cpu_info.p_core_workers);
    let cpu_sched = cpu::build_schedule(cpu_info.p_core_workers.max(1));
    let detect_parallel = cpu::detected_parallelism();

    let mut brand_buf = [0u8; 96];
    let brand_len =
        sysctl_string(b"machdep.cpu.brand_string\0", &mut brand_buf).unwrap_or(0);
    let brand_str = core::str::from_utf8(&brand_buf[..brand_len]).unwrap_or("");
    let is_apple_silicon =
        contains_ascii_nocase(brand_str, "apple") || contains_ascii_nocase(brand_str, "arm");
    let hint_ncpu = sysctl_u32(b"hw.ncpu\0").unwrap_or(0);
    let hint_memsize = sysctl_u64(b"hw.memsize\0").unwrap_or(0);

    let ram_cfg = ram::default_config();
    let ram_clamp = ram::clamp_workers(ram_cfg.page_size.max(1));
    let ram_sched = ram::build_schedule(ram_cfg.page_size.max(1));

    tpu::configure_device(0, 0, family);
    let probed = tpu::probe_device();
    let tpu_cfg = tpu::default_config();
    let tpu_clamp = tpu::clamp_workers(tpu_cfg.compute_queues).max(1);
    let tpu_sched = tpu::build_schedule(tpu_cfg.workgroup_size.max(1));

    let diag = tpu::run_diagnostics();
    let init_flag = tpu::device_is_initialized() as usize;
    let mmio_base = tpu::device_mmio_base();
    let status = tpu::read_status_reg();
    let reg0 = tpu::read_reg(0);
    tpu::write_reg(0, reg0);
    let cmd_status = tpu::submit_compute(0, 0, 0);
    tpu::power_cycle();
    tpu::install_irq(family, 0);
    let streams = tpu::smmu_active_streams();
    let mapped_dma = tpu::smmu_map_dma(mmio_base, page);
    tpu::smmu_set_attrs(0, family);
    let mapped_bytes = tpu::smmu_mapped_bytes();
    let region = tpu::dma_region_base();
    let dma_buf = tpu::dma_alloc(page).unwrap_or(region);
    let dma_done = tpu::dma_submit(dma_buf as u64, page as u32) as usize;
    let dma_ready = tpu::dma_complete() as usize;
    tpu::dma_clean_invalidate(dma_buf, page);

    let mem_total = if hint_memsize > 0 {
        hint_memsize
    } else {
        unified_memory_bytes().max(DEFAULT_TPU_TOTAL_MEMORY_BYTES)
    };
    let mem_avail = mem_total - (mem_total >> 3);
    let flops = if is_apple_silicon {
        DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND
    } else {
        DEFAULT_TPU_SUSTAINED_FLOPS_PER_SECOND / 2
    };

    let seed = (diag ^ init_flag ^ mapped_dma ^ mapped_bytes ^ region ^ dma_buf)
        .wrapping_add(status as usize)
        .wrapping_add(reg0 as usize)
        .wrapping_add(cmd_status as usize)
        .wrapping_add(streams)
        .wrapping_add(dma_done)
        .wrapping_add(dma_ready)
        .wrapping_add(family as usize)
        .wrapping_add(logical)
        .wrapping_add(page)
        .wrapping_add(detect_parallel)
        .wrapping_add(cpu_clamp)
        .wrapping_add(cpu_sched.chunks)
        .wrapping_add(cpu_sched.chunk_size)
        .wrapping_add(cpu_sched.frame_budget_us as usize)
        .wrapping_add(cpu_info.l2_cache_bytes as usize)
        .wrapping_add(cpu_info.l3_cache_bytes as usize)
        .wrapping_add(cpu_info.e_core_workers)
        .wrapping_add(cpu_info.render_workers)
        .wrapping_add(cpu_info.freq_p_max_hz as usize)
        .wrapping_add(cpu_info.freq_e_max_hz as usize)
        .wrapping_add(cpu_info.frame_budget_us as usize)
        .wrapping_add(cpu_info.low_power as usize)
        .wrapping_add(ram_clamp)
        .wrapping_add(ram_sched.chunks)
        .wrapping_add(ram_sched.chunk_size)
        .wrapping_add(ram_sched.frame_budget_us as usize)
        .wrapping_add(ram_cfg.page_size)
        .wrapping_add(ram_cfg.total_bytes as usize)
        .wrapping_add(ram_cfg.available_bytes.unwrap_or(0) as usize)
        .wrapping_add(ram_cfg.frame_budget_us as usize)
        .wrapping_add(ram_cfg.low_power as usize)
        .wrapping_add(tpu_clamp)
        .wrapping_add(tpu_sched.chunks)
        .wrapping_add(tpu_sched.chunk_size)
        .wrapping_add(tpu_sched.frame_budget_us as usize)
        .wrapping_add(tpu_cfg.render_threads)
        .wrapping_add(tpu_cfg.double_buffered as usize)
        .wrapping_add(tpu_cfg.frame_budget_us as usize)
        .wrapping_add(tpu_cfg.low_power as usize)
        .wrapping_add(probed as usize)
        .wrapping_add(hint_ncpu as usize);

    register_tpu_command_handler_with_hardware_profile(
        tpu_command_handler,
        seed,
        ALL_OPCODE_MASK,
        flops as u128,
        mem_total as u128,
        mem_avail as u128,
    );
}

pub fn tpu_dispatch_count() -> usize {
    TPU_DISPATCH_TOTAL.load(Ordering::Relaxed)
}

pub fn tpu_last_status() -> u32 {
    TPU_LAST_STATUS.load(Ordering::Relaxed)
}

pub fn tpu_last_opcode() -> u32 {
    TPU_LAST_OPCODE.load(Ordering::Relaxed)
}
