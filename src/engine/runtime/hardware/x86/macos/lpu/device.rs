use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static LPU_MMIO_BASE: AtomicUsize = AtomicUsize::new(0);
static LPU_MMIO_SIZE: AtomicUsize = AtomicUsize::new(0);
static LPU_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub struct ArmLpuContext {
    pub mmio_base: usize,
    pub mmio_size: usize,
    pub device_id: u32,
    pub spi_id: u32,
    pub smmu_stream_id: u32,
    pub dma_region: usize,
}

pub fn init_lpu(mmio_base: usize, mmio_size: usize, spi_id: u32) -> Option<ArmLpuContext> {
    let device_id = super::platform::read_device_id(mmio_base);
    if device_id == 0 || device_id == 0xFFFF_FFFF {
        return None;
    }

    LPU_MMIO_BASE.store(mmio_base, Ordering::Release);
    LPU_MMIO_SIZE.store(mmio_size, Ordering::Release);

    super::platform::reset_device(mmio_base);
    super::platform::enable_clocks(mmio_base);

    let stream_id = super::smmu::configure_lpu_stream(mmio_base, 0x300);
    super::smmu::set_lpu_attributes(
        stream_id,
        super::smmu::ATTR_CACHEABLE | super::smmu::ATTR_SHAREABLE,
    );

    super::platform::configure_gic_spi(spi_id, 0);

    let dma_region = super::dma::setup_inference_region(mmio_base);

    LPU_INITIALIZED.store(true, Ordering::Release);

    Some(ArmLpuContext {
        mmio_base,
        mmio_size,
        device_id,
        spi_id,
        smmu_stream_id: stream_id,
        dma_region,
    })
}

pub fn is_initialized() -> bool {
    LPU_INITIALIZED.load(Ordering::Acquire)
}

pub fn read_lpu_reg(offset: usize) -> u32 {
    let base = LPU_MMIO_BASE.load(Ordering::Acquire);
    if base == 0 {
        return 0;
    }
    unsafe { super::super::mmio::mmio_read32(base + offset) }
}

pub fn write_lpu_reg(offset: usize, val: u32) {
    let base = LPU_MMIO_BASE.load(Ordering::Acquire);
    if base != 0 {
        unsafe {
            super::super::mmio::mmio_write32(base + offset, val);
        }
    }
}

pub fn diagnostics(mmio_base: usize) -> usize {
    let mut sig = is_initialized() as usize;
    sig ^= read_lpu_reg(0) as usize;
    write_lpu_reg(0, read_lpu_reg(4));
    sig ^= super::platform::read_revision(mmio_base) as usize;
    super::platform::power_on(mmio_base);
    super::platform::power_off(mmio_base);
    sig ^= super::platform::is_powered() as usize;
    super::platform::configure_inference(mmio_base, 0, 0, 0);
    super::platform::submit_prefill(mmio_base, 0, 0);
    super::platform::submit_decode(mmio_base, 0);
    super::platform::submit_speculative(mmio_base, 0, 0);
    sig ^= super::platform::is_ready(mmio_base) as usize;
    sig ^= super::platform::is_busy(mmio_base) as usize;
    sig ^= super::platform::has_error(mmio_base) as usize;
    sig ^= super::smmu::ATTR_NON_CACHEABLE as usize;
    sig ^= super::smmu::active_lpu_streams() as usize;
    sig ^= super::dma::alloc_inference_buffer(0);
    super::dma::submit_inference_dma(mmio_base, 0, 0, 0, true);
    sig ^= super::dma::is_dma_complete(mmio_base) as usize;
    sig ^= super::dma::has_dma_error(mmio_base) as usize;
    sig ^= super::dma::is_region_initialized() as usize;
    sig ^= super::dma::remaining_capacity();
    sig
}
