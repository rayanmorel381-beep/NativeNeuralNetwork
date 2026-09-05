use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub(super) mod backend;
pub(super) mod scheduler;

pub mod device;
pub mod dma;
pub mod platform;
pub mod smmu;

static TPU_PROBE_BASE: AtomicUsize = AtomicUsize::new(0);
static TPU_PROBE_SIZE: AtomicUsize = AtomicUsize::new(0);
static TPU_PROBE_SPI: AtomicU32 = AtomicU32::new(0);

pub(crate) fn configure_device(mmio_base: usize, mmio_size: usize, spi_id: u32) {
    TPU_PROBE_BASE.store(mmio_base, Ordering::Release);
    TPU_PROBE_SIZE.store(mmio_size, Ordering::Release);
    TPU_PROBE_SPI.store(spi_id, Ordering::Release);
    if mmio_base > 0 {
        platform::set_aic_base(mmio_base);
    }
}

pub(crate) fn probe_device() -> bool {
    let base = TPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return false;
    }
    let size = TPU_PROBE_SIZE.load(Ordering::Acquire);
    let spi = TPU_PROBE_SPI.load(Ordering::Acquire);
    if let Some(ctx) = device::init_tpu(base, size, spi) {
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
    let base = TPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return 0;
    }
    let diag = device::diagnostics(base);
    let hi = unsafe { super::mmio::mmio_read64(base + 8) } as usize;
    unsafe { super::mmio::mmio_write64(base + 8, diag as u64) };
    unsafe { super::mmio::dsb_sy() };
    unsafe { super::mmio::isb() };
    diag ^ hi
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
    device::mmio_base()
}

pub(crate) fn device_is_initialized() -> bool {
    device::is_initialized()
}

pub(crate) fn read_status_reg() -> u32 {
    let base = device::mmio_base();
    if base == 0 {
        return 0;
    }
    platform::read_status(base)
}

pub(crate) fn read_reg(offset: usize) -> u32 {
    device::read_tpu_reg(offset)
}

pub(crate) fn write_reg(offset: usize, val: u32) {
    device::write_tpu_reg(offset, val);
}

pub(crate) fn submit_compute(cmd: u32, data_addr: u32, size: u32) -> u32 {
    let base = device::mmio_base();
    if base == 0 {
        return 0;
    }
    platform::submit_compute(base, cmd, data_addr, size);
    platform::enable_interrupts(base);
    platform::clear_interrupts(base)
}

pub(crate) fn power_cycle() {
    let base = device::mmio_base();
    if base == 0 {
        return;
    }
    platform::power_off(base);
    platform::power_on(base);
}

pub(crate) fn install_irq(spi_id: u32, target_cpu: u32) {
    platform::configure_aic_irq(spi_id, target_cpu);
}

pub(crate) fn smmu_active_streams() -> usize {
    smmu::stream_count()
}

pub(crate) fn smmu_map_dma(phys: usize, size: usize) -> usize {
    smmu::map_tpu_dma(phys, size)
}

pub(crate) fn smmu_set_attrs(stream_id: u32, attrs: u32) {
    smmu::set_tpu_attributes(stream_id, attrs);
}

pub(crate) fn smmu_mapped_bytes() -> usize {
    smmu::mapped_bytes()
}

pub(crate) fn dma_alloc(size: usize) -> Option<usize> {
    dma::alloc_dma_buffer(size)
}

pub(crate) fn dma_submit(src_phys: u64, size: u32) -> bool {
    dma::submit_dma_transfer(src_phys, size)
}

pub(crate) fn dma_complete() -> bool {
    dma::is_dma_complete()
}

pub(crate) fn dma_region_base() -> usize {
    dma::dma_region_base()
}

pub(crate) fn dma_clean_invalidate(va: usize, size: usize) {
    dma::clean_and_invalidate(va, size);
}
