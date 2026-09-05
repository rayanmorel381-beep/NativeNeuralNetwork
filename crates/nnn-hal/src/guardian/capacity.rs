use core::sync::atomic::{AtomicU64, Ordering};

static CPU_READER: AtomicU64 = AtomicU64::new(0);
static MEMORY_READER: AtomicU64 = AtomicU64::new(0);
static SWAP_READER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_cpu_reader(reader: fn() -> u64) {
    CPU_READER.store(reader as usize as u64, Ordering::Release);
}

pub(crate) fn set_memory_reader(reader: fn() -> u64) {
    MEMORY_READER.store(reader as usize as u64, Ordering::Release);
}

pub(crate) fn set_swap_reader(reader: fn() -> u64) {
    SWAP_READER.store(reader as usize as u64, Ordering::Release);
}

fn read(reader: &AtomicU64) -> Option<u64> {
    let raw = reader.load(Ordering::Acquire);
    if raw == 0 { return None; }
    let reader: fn() -> u64 = unsafe { core::mem::transmute(raw as usize) };
    Some(reader())
}

pub(crate) fn cpu_usage() -> Option<u64> { read(&CPU_READER) }
pub(crate) fn memory_usage() -> Option<u64> { read(&MEMORY_READER) }
pub(crate) fn swap_usage() -> Option<u64> { read(&SWAP_READER) }
