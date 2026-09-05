use core::sync::atomic::{AtomicUsize, Ordering};

const DMA_BUF_REG: usize = 0x200;
const DMA_SIZE_REG: usize = 0x208;
const DMA_CTRL_REG: usize = 0x210;
const DMA_STATUS_REG: usize = 0x214;
const DMA_REGION_SIZE: usize = 16 * 1024 * 1024;
const DMA_GRANULE: usize = 65536;

static DMA_REGION_BASE: AtomicUsize = AtomicUsize::new(0);
static DMA_OFFSET: AtomicUsize = AtomicUsize::new(0);

pub fn setup_dma_region(mmio_base: usize) -> usize {
    let phys = (mmio_base + DMA_REGION_SIZE) & !(DMA_GRANULE - 1);

    DMA_REGION_BASE.store(phys, Ordering::Release);

    unsafe {
        super::super::mmio::mmio_write32(mmio_base + DMA_BUF_REG, phys as u32);
        super::super::mmio::mmio_write32(mmio_base + DMA_BUF_REG + 4, (phys >> 32) as u32);
        super::super::mmio::mmio_write32(mmio_base + DMA_SIZE_REG, DMA_REGION_SIZE as u32);
        super::super::mmio::mmio_write32(mmio_base + DMA_CTRL_REG, 1);
    }

    phys
}

pub fn alloc_dma_buffer(size: usize) -> Option<usize> {
    let base = DMA_REGION_BASE.load(Ordering::Acquire);
    if base == 0 {
        return None;
    }
    let aligned = (size + DMA_GRANULE - 1) & !(DMA_GRANULE - 1);
    let off = DMA_OFFSET.fetch_add(aligned, Ordering::AcqRel);
    if off + aligned > DMA_REGION_SIZE {
        return None;
    }
    Some(base + off)
}

pub fn clean_and_invalidate(va_start: usize, size: usize) {
    let line_size = 64usize;
    let mut addr = va_start & !(line_size - 1);
    let end = va_start + size;
    while addr < end {
        unsafe {
            core::ptr::write_volatile(
                addr as *mut u8,
                core::ptr::read_volatile(addr as *const u8),
            );
        }
        addr += line_size;
    }
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

pub fn submit_dma_transfer(src_phys: u64, size: u32) -> bool {
    let base = super::device::mmio_base();
    if base == 0 {
        return false;
    }
    unsafe {
        super::super::mmio::mmio_write32(base + DMA_BUF_REG, src_phys as u32);
        super::super::mmio::mmio_write32(base + DMA_BUF_REG + 4, (src_phys >> 32) as u32);
        super::super::mmio::mmio_write32(base + DMA_SIZE_REG, size);
        super::super::mmio::mmio_write32(base + DMA_CTRL_REG, 0x03);
    }
    true
}

pub fn is_dma_complete() -> bool {
    let base = super::device::mmio_base();
    if base == 0 {
        return false;
    }
    let status = unsafe { super::super::mmio::mmio_read32(base + DMA_STATUS_REG) };
    status & 0x01 != 0
}

pub fn dma_region_base() -> usize {
    DMA_REGION_BASE.load(Ordering::Acquire)
}
