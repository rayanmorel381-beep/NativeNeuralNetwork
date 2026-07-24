use core::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

pub(super) mod backend;
pub(super) mod scheduler;

pub mod device;
pub mod dma;
pub mod platform;
pub mod smmu;

static TPU_PROBE_BASE: AtomicUsize = AtomicUsize::new(0);
static TPU_PROBE_SIZE: AtomicUsize = AtomicUsize::new(0);
static TPU_PROBE_SPI: AtomicU32 = AtomicU32::new(0);
static TPU_POWER_ACTIVE: AtomicBool = AtomicBool::new(false);

pub(crate) fn configure_device(mmio_base: usize, mmio_size: usize, spi_id: u32) {
    TPU_PROBE_BASE.store(mmio_base, Ordering::Release);
    TPU_PROBE_SIZE.store(mmio_size, Ordering::Release);
    TPU_PROBE_SPI.store(spi_id, Ordering::Release);
    if mmio_base > 0 {
        platform::power_on(mmio_base);
        platform::enable_clocks(mmio_base);
        TPU_POWER_ACTIVE.store(true, Ordering::Release);
    }
}

pub(crate) fn probe_device() -> bool {
    let base = TPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return false;
    }
    let size = TPU_PROBE_SIZE.load(Ordering::Acquire);
    let spi = TPU_PROBE_SPI.load(Ordering::Acquire);
    device::init_tpu(base, size, spi).map(|ctx| {
        ctx.mmio_base > 0
            && ctx.mmio_size > 0
            && ctx.device_id > 0
            && ctx.spi_id > 0
            && ctx.smmu_stream_id > 0
            && ctx.dma_region > 0
    }).unwrap_or(false)
}

pub(crate) fn run_diagnostics() -> usize {
    let base = TPU_PROBE_BASE.load(Ordering::Acquire);
    if base == 0 {
        return 0;
    }
    device::diagnostics(base)
}

pub(crate) fn default_config() -> backend::VendorBackendConfig {
    backend::default_backend_config()
}

pub(crate) fn build_schedule(work_items: usize) -> scheduler::VendorSchedule {
    let _ = scheduler::recommended_chunk_size(work_items);
    scheduler::build_schedule(work_items)
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    backend::clamp_workers(requested)
}
