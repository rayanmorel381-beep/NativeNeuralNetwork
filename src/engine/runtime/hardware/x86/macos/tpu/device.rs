use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static TPU_MMIO_BASE: AtomicUsize = AtomicUsize::new(0);
static TPU_MMIO_SIZE: AtomicUsize = AtomicUsize::new(0);
static TPU_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub struct ArmTpuContext {
    pub mmio_base: usize,
    pub mmio_size: usize,
    pub device_id: u32,
    pub spi_id: u32,
    pub smmu_stream_id: u32,
    pub dma_region: usize,
}

pub fn init_tpu(mmio_base: usize, mmio_size: usize, spi_id: u32) -> Option<ArmTpuContext> {
    let device_id = super::platform::read_device_id(mmio_base);
    if device_id == 0 || device_id == 0xFFFF_FFFF {
        return None;
    }

    TPU_MMIO_BASE.store(mmio_base, Ordering::Release);
    TPU_MMIO_SIZE.store(mmio_size, Ordering::Release);

    super::platform::reset_device(mmio_base);
    super::platform::enable_clocks(mmio_base);

    let stream_id = super::smmu::configure_tpu_stream(mmio_base, 0x200);
    super::smmu::set_tpu_attributes(
        stream_id,
        super::smmu::ATTR_NON_CACHEABLE | super::smmu::ATTR_DEVICE,
    );

    super::platform::configure_gic_spi(spi_id, 0);

    let dma_region = super::dma::setup_dma_region(mmio_base);

    TPU_INITIALIZED.store(true, Ordering::Release);

    Some(ArmTpuContext {
        mmio_base,
        mmio_size,
        device_id,
        spi_id,
        smmu_stream_id: stream_id,
        dma_region,
    })
}

pub fn is_initialized() -> bool {
    TPU_INITIALIZED.load(Ordering::Acquire)
}

pub(super) fn mmio_base() -> usize {
    TPU_MMIO_BASE.load(Ordering::Acquire)
}

pub fn read_tpu_reg(offset: usize) -> u32 {
    let base = TPU_MMIO_BASE.load(Ordering::Acquire);
    if base == 0 {
        return 0;
    }
    unsafe { super::super::mmio::mmio_read32(base + offset) }
}

pub fn write_tpu_reg(offset: usize, val: u32) {
    let base = TPU_MMIO_BASE.load(Ordering::Acquire);
    if base != 0 {
        unsafe {
            super::super::mmio::mmio_write32(base + offset, val);
        }
    }
}

pub fn diagnostics(mmio_base: usize) -> usize {
    let mut sig = is_initialized() as usize;
    sig ^= read_tpu_reg(0) as usize;
    write_tpu_reg(0, read_tpu_reg(4));
    super::platform::submit_compute(mmio_base, 0, 0, 0);
    sig ^= super::platform::read_status(mmio_base) as usize;
    super::platform::enable_interrupts(mmio_base);
    sig ^= super::platform::clear_interrupts(mmio_base) as usize;
    super::platform::power_on(mmio_base);
    super::platform::power_off(mmio_base);
    sig ^= super::smmu::ATTR_READ as usize ^ super::smmu::ATTR_WRITE as usize;
    sig ^= super::smmu::map_tpu_dma(0, 0);
    sig ^= super::smmu::stream_count();
    sig ^= super::dma::alloc_dma_buffer(0).unwrap_or(0);
    super::dma::clean_and_invalidate(mmio_base, 64);
    sig ^= super::dma::submit_dma_transfer(0, 0) as usize;
    sig ^= super::dma::is_dma_complete() as usize;
    sig ^= super::dma::dma_region_base();
    sig
}
