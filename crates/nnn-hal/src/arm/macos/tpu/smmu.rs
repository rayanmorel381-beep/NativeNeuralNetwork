use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub const ATTR_NON_CACHEABLE: u32 = 1 << 0;
pub const ATTR_DEVICE: u32 = 1 << 1;
pub const ATTR_READ: u32 = 1 << 2;
pub const ATTR_WRITE: u32 = 1 << 3;

const MAX_STREAMS: usize = 8;

const DART_TCR_BASE: usize = 0x100;
const DART_TCR_BYPASS: u32 = 1 << 8;
const DART_TCR_TRANSLATE_ENABLE: u32 = 1 << 1;

static STREAM_COUNT: AtomicUsize = AtomicUsize::new(0);
static STREAM_IDS: [AtomicU32; MAX_STREAMS] = [const { AtomicU32::new(0) }; MAX_STREAMS];
static STREAM_ATTRS: [AtomicU32; MAX_STREAMS] = [const { AtomicU32::new(0) }; MAX_STREAMS];
static DART_BASE: AtomicUsize = AtomicUsize::new(0);

pub fn configure_tpu_stream(device_base: usize, stream_id: u32) -> u32 {
    let idx = STREAM_COUNT.fetch_add(1, Ordering::AcqRel);
    if idx >= MAX_STREAMS {
        STREAM_COUNT.fetch_sub(1, Ordering::Release);
        return 0;
    }

    STREAM_IDS[idx].store(stream_id, Ordering::Release);
    STREAM_ATTRS[idx].store(ATTR_NON_CACHEABLE | ATTR_DEVICE, Ordering::Release);
    DART_BASE.store(device_base, Ordering::Release);

    let tcr_addr = device_base + DART_TCR_BASE + idx * 4;
    unsafe {
        super::super::mmio::mmio_write32(tcr_addr, DART_TCR_BYPASS);
    }

    stream_id
}

pub fn set_tpu_attributes(stream_id: u32, attrs: u32) {
    for i in 0..MAX_STREAMS {
        if STREAM_IDS[i].load(Ordering::Acquire) == stream_id {
            STREAM_ATTRS[i].store(attrs, Ordering::Release);

            let dart_base = DART_BASE.load(Ordering::Acquire);
            if dart_base != 0 {
                let tcr = if attrs & (ATTR_READ | ATTR_WRITE) != 0 {
                    DART_TCR_TRANSLATE_ENABLE
                } else {
                    DART_TCR_BYPASS
                };
                let tcr_addr = dart_base + DART_TCR_BASE + i * 4;
                unsafe {
                    super::super::mmio::mmio_write32(tcr_addr, tcr);
                }
            }
            return;
        }
    }
}

static MAPPED_BYTES: AtomicUsize = AtomicUsize::new(0);

pub fn map_tpu_dma(phys: usize, size: usize) -> usize {
    MAPPED_BYTES.fetch_add(size, Ordering::AcqRel);
    phys
}

pub fn stream_count() -> usize {
    STREAM_COUNT.load(Ordering::Acquire)
}

pub fn mapped_bytes() -> usize {
    MAPPED_BYTES.load(Ordering::Acquire)
}
