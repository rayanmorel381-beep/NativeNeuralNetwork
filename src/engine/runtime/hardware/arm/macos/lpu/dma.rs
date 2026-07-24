use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const INFERENCE_REGION_SIZE: usize = 4 * 1024 * 1024;
const INFERENCE_GRANULE: usize = 16 * 1024;

const REG_DMA_SRC: usize = 0x0200;
const REG_DMA_DST: usize = 0x0204;
const REG_DMA_LEN: usize = 0x0208;
const REG_DMA_CTRL: usize = 0x020C;
const REG_DMA_STATUS: usize = 0x0210;

const DMA_START: u32 = 1 << 0;
const DMA_DIRECTION_TO_DEVICE: u32 = 1 << 1;
const DMA_DIRECTION_FROM_DEVICE: u32 = 1 << 2;

const DMA_STATUS_COMPLETE: u32 = 1 << 0;
const DMA_STATUS_ERROR: u32 = 1 << 4;

static DMA_REGION_BASE: AtomicUsize = AtomicUsize::new(0);
static DMA_REGION_OFFSET: AtomicUsize = AtomicUsize::new(0);
static DMA_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub fn setup_inference_region(mmio_base: usize) -> usize {
    let phys = (mmio_base + INFERENCE_REGION_SIZE) & !(INFERENCE_GRANULE - 1);

    DMA_REGION_BASE.store(phys, Ordering::Release);
    DMA_REGION_OFFSET.store(0, Ordering::Release);

    let iova = super::smmu::map_lpu_dma(phys, INFERENCE_REGION_SIZE);

    unsafe {
        super::super::mmio::mmio_write32(mmio_base + REG_DMA_SRC, iova as u32);
    }

    DMA_INITIALIZED.store(true, Ordering::Release);
    phys
}

pub fn alloc_inference_buffer(size: usize) -> usize {
    let aligned = (size + INFERENCE_GRANULE - 1) & !(INFERENCE_GRANULE - 1);
    let offset = DMA_REGION_OFFSET.fetch_add(aligned, Ordering::AcqRel);

    if offset + aligned > INFERENCE_REGION_SIZE {
        DMA_REGION_OFFSET.fetch_sub(aligned, Ordering::AcqRel);
        return 0;
    }

    let base = DMA_REGION_BASE.load(Ordering::Acquire);
    let addr = base + offset;

    clean_and_invalidate_range(addr, aligned);
    addr
}

pub fn submit_inference_dma(mmio_base: usize, src: u32, dst: u32, len: u32, to_device: bool) {
    clean_and_invalidate_range(src as usize, len as usize);

    unsafe {
        super::super::mmio::mmio_write32(mmio_base + REG_DMA_SRC, src);
        super::super::mmio::mmio_write32(mmio_base + REG_DMA_DST, dst);
        super::super::mmio::mmio_write32(mmio_base + REG_DMA_LEN, len);

        let direction = if to_device {
            DMA_DIRECTION_TO_DEVICE
        } else {
            DMA_DIRECTION_FROM_DEVICE
        };
        super::super::mmio::mmio_write32(mmio_base + REG_DMA_CTRL, DMA_START | direction);
    }
}

pub fn is_dma_complete(mmio_base: usize) -> bool {
    unsafe {
        let status = super::super::mmio::mmio_read32(mmio_base + REG_DMA_STATUS);
        status & DMA_STATUS_COMPLETE != 0
    }
}

pub fn has_dma_error(mmio_base: usize) -> bool {
    unsafe {
        let status = super::super::mmio::mmio_read32(mmio_base + REG_DMA_STATUS);
        status & DMA_STATUS_ERROR != 0
    }
}

fn clean_and_invalidate_range(addr: usize, size: usize) {
    let cache_line_size: usize = 64;
    let start = addr & !(cache_line_size - 1);
    let end = addr + size;
    let mut current = start;
    while current < end {
        unsafe {
            core::ptr::write_volatile(current as *mut u8, core::ptr::read_volatile(current as *const u8));
        }
        current += cache_line_size;
    }
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

pub fn is_region_initialized() -> bool {
    DMA_INITIALIZED.load(Ordering::Acquire)
}

pub fn remaining_capacity() -> usize {
    let offset = DMA_REGION_OFFSET.load(Ordering::Acquire);
    INFERENCE_REGION_SIZE.saturating_sub(offset)
}
