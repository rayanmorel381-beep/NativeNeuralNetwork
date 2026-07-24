use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub(super) mod backend;
pub(super) mod scheduler;

pub mod device;
pub mod dma;
pub mod platform;
pub mod smmu;

static LPU_PROBE_BASE: AtomicUsize = AtomicUsize::new(0);
static LPU_PROBE_SIZE: AtomicUsize = AtomicUsize::new(0);
static LPU_PROBE_SPI: AtomicU32 = AtomicU32::new(0);

pub(crate) fn configure_device(mmio_base: usize, mmio_size: usize, spi_id: u32) {
    LPU_PROBE_BASE.store(mmio_base, Ordering::Release);
    LPU_PROBE_SIZE.store(mmio_size, Ordering::Release);
    LPU_PROBE_SPI.store(spi_id, Ordering::Release);
    if mmio_base > 0 {
        platform::set_aic_base(mmio_base);
    }
}

pub(crate) fn probe_device() -> bool {
    let base = LPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return false;
    }
    let size = LPU_PROBE_SIZE.load(Ordering::Acquire);
    let spi = LPU_PROBE_SPI.load(Ordering::Acquire);
    if let Some(ctx) = device::init_lpu(base, size, spi) {
        ctx.mmio_base > 0
            && ctx.mmio_size > 0
            && ctx.device_id > 0
            && ctx.spi_id > 0
            && ctx.smmu_stream_id > 0
            && ctx.dma_region > 0
    } else {
        false
    }
}

pub(crate) fn run_diagnostics() -> usize {
    let base = LPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return 0;
    }
    let diag = device::diagnostics(base);
    let attr_sig = smmu::ATTR_DEVICE as usize;
    let hi = unsafe { super::mmio::mmio_read64(base + 8) } as usize;
    unsafe { super::mmio::mmio_write64(base + 8, diag as u64) };
    unsafe { super::mmio::dsb_sy() };
    diag ^ hi ^ attr_sig
}

pub(crate) fn default_config() -> backend::VendorBackendConfig {
    backend::default_backend_config()
}

pub(crate) fn build_schedule(work_items: usize) -> scheduler::VendorSchedule {
    scheduler::build_schedule(work_items)
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    backend::clamp_workers(requested)
}

pub(crate) fn device_mmio_base() -> usize {
    let base = LPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 || !device::is_initialized() {
        return 0;
    }
    base
}

pub(crate) fn device_is_initialized() -> bool {
    device::is_initialized()
}

pub(crate) fn read_status_reg() -> u32 {
    let base = device_mmio_base();
    if base == 0 {
        return 0;
    }
    let ready = platform::is_ready(base) as u32;
    let busy = platform::is_busy(base) as u32;
    let err = platform::has_error(base) as u32;
    (ready << 0) | (busy << 1) | (err << 4)
}

pub(crate) fn read_reg(offset: usize) -> u32 {
    device::read_lpu_reg(offset)
}

pub(crate) fn write_reg(offset: usize, val: u32) {
    device::write_lpu_reg(offset, val);
}

pub(crate) fn submit_prefill(token_addr: u32, token_count: u32) -> bool {
    let base = device_mmio_base();
    if base == 0 {
        return false;
    }
    platform::configure_inference(base, 0, token_count, 0);
    platform::submit_prefill(base, token_addr, token_count);
    true
}

pub(crate) fn submit_decode(token_addr: u32) -> bool {
    let base = device_mmio_base();
    if base == 0 {
        return false;
    }
    platform::submit_decode(base, token_addr);
    true
}

pub(crate) fn submit_speculative(token_addr: u32, draft_count: u32) -> bool {
    let base = device_mmio_base();
    if base == 0 {
        return false;
    }
    platform::submit_speculative(base, token_addr, draft_count);
    true
}

pub(crate) fn power_cycle() {
    let base = LPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return;
    }
    platform::power_off(base);
    platform::power_on(base);
}

pub(crate) fn is_powered() -> bool {
    platform::is_powered()
}

pub(crate) fn install_irq(spi_id: u32, target_cpu: u32) {
    platform::configure_aic_irq(spi_id, target_cpu);
}

pub(crate) fn revision() -> u32 {
    let base = device_mmio_base();
    if base == 0 {
        return 0;
    }
    platform::read_revision(base)
}

pub(crate) fn device_id() -> u32 {
    let base = device_mmio_base();
    if base == 0 {
        return 0;
    }
    platform::read_device_id(base)
}

pub(crate) fn dma_alloc(size: usize) -> usize {
    dma::alloc_inference_buffer(size)
}

pub(crate) fn dma_submit(src: u32, dst: u32, len: u32, to_device: bool) -> bool {
    let base = device_mmio_base();
    if base == 0 {
        return false;
    }
    dma::submit_inference_dma(base, src, dst, len, to_device);
    true
}

pub(crate) fn dma_complete() -> bool {
    let base = device_mmio_base();
    if base == 0 {
        return false;
    }
    dma::is_dma_complete(base)
}

pub(crate) fn dma_error() -> bool {
    let base = device_mmio_base();
    if base == 0 {
        return false;
    }
    dma::has_dma_error(base)
}

pub(crate) fn dma_region_ready() -> bool {
    dma::is_region_initialized()
}

pub(crate) fn dma_remaining() -> usize {
    dma::remaining_capacity()
}

pub(crate) fn smmu_streams() -> u32 {
    smmu::active_lpu_streams()
}

pub(crate) fn smmu_map(phys: usize, size: usize) -> usize {
    smmu::map_lpu_dma(phys, size)
}

pub(crate) fn smmu_set_attrs(stream_id: u32, attrs: u32) {
    smmu::set_lpu_attributes(stream_id, attrs);
}

pub(crate) fn smmu_mapped_bytes() -> usize {
    smmu::mapped_bytes()
}
