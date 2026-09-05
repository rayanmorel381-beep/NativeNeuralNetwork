use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub const ATTR_NON_CACHEABLE: u32 = 1 << 0;
pub const ATTR_DEVICE: u32 = 1 << 1;
pub const ATTR_READ: u32 = 1 << 2;
pub const ATTR_WRITE: u32 = 1 << 3;

const MAX_STREAMS: usize = 8;

static STREAM_COUNT: AtomicUsize = AtomicUsize::new(0);
static STREAM_IDS: [AtomicU32; MAX_STREAMS] = [const { AtomicU32::new(0) }; MAX_STREAMS];
static STREAM_ATTRS: [AtomicU32; MAX_STREAMS] = [const { AtomicU32::new(0) }; MAX_STREAMS];

pub fn configure_tpu_stream(device_base: usize, stream_id: u32) -> u32 {
    let idx = STREAM_COUNT.fetch_add(1, Ordering::AcqRel);
    if idx >= MAX_STREAMS {
        STREAM_COUNT.fetch_sub(1, Ordering::Release);
        return 0;
    }

    STREAM_IDS[idx].store(stream_id, Ordering::Release);

    STREAM_ATTRS[idx].store(device_base as u32, Ordering::Release);

    stream_id
}

pub fn set_tpu_attributes(stream_id: u32, attrs: u32) {
    for i in 0..MAX_STREAMS {
        if STREAM_IDS[i].load(Ordering::Acquire) == stream_id {
            STREAM_ATTRS[i].store(attrs, Ordering::Release);
            return;
        }
    }
}

pub fn map_tpu_dma(phys: usize, size: usize) -> usize {
    let _ = size;
    phys
}

pub fn stream_count() -> usize {
    STREAM_COUNT.load(Ordering::Acquire)
}
