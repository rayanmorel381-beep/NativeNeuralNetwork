use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

const MAX_LPU_STREAMS: usize = 8;

pub const ATTR_CACHEABLE: u32 = 1 << 0;
pub const ATTR_SHAREABLE: u32 = 1 << 1;
pub const ATTR_NON_CACHEABLE: u32 = 1 << 2;
pub const ATTR_DEVICE: u32 = 1 << 3;

static STREAM_COUNT: AtomicU32 = AtomicU32::new(0);
static STREAM_IDS: [AtomicU32; MAX_LPU_STREAMS] = [
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
];
static STREAM_ATTRS: [AtomicU32; MAX_LPU_STREAMS] = [
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
    AtomicU32::new(0),
];
static SMMU_BASE: AtomicUsize = AtomicUsize::new(0);

const SMMU_CBAR_OFFSET: usize = 0x1000;

pub fn configure_lpu_stream(mmio_base: usize, stream_id_base: u32) -> u32 {
    let idx = STREAM_COUNT.fetch_add(1, Ordering::AcqRel) as usize;
    if idx >= MAX_LPU_STREAMS {
        STREAM_COUNT.fetch_sub(1, Ordering::AcqRel);
        return 0;
    }

    let stream_id = stream_id_base + idx as u32;
    STREAM_IDS[idx].store(stream_id, Ordering::Release);

    STREAM_ATTRS[idx].store(mmio_base as u32, Ordering::Release);
    SMMU_BASE.store(mmio_base, Ordering::Release);

    stream_id
}

pub fn set_lpu_attributes(stream_id: u32, attrs: u32) {
    let count = STREAM_COUNT.load(Ordering::Acquire) as usize;
    for i in 0..count {
        if STREAM_IDS[i].load(Ordering::Acquire) == stream_id {
            STREAM_ATTRS[i].store(attrs, Ordering::Release);

            let smmu_base = SMMU_BASE.load(Ordering::Acquire);
            if smmu_base != 0 {
                let cbar_addr = smmu_base + SMMU_CBAR_OFFSET + i * 4;
                let mut cbar_val: u32 = 0x01;
                if attrs & ATTR_CACHEABLE != 0 {
                    cbar_val |= 0x07 << 16;
                }
                if attrs & ATTR_SHAREABLE != 0 {
                    cbar_val |= 0x03 << 22;
                }
                if attrs & ATTR_DEVICE != 0 {
                    cbar_val |= 0x04 << 16;
                }
                unsafe {
                    super::super::mmio::mmio_write32(cbar_addr, cbar_val);
                }
            }
            break;
        }
    }
}

pub fn map_lpu_dma(phys: usize, size: usize) -> usize {
    let _ = size;
    phys
}

pub fn active_lpu_streams() -> u32 {
    STREAM_COUNT.load(Ordering::Acquire)
}
