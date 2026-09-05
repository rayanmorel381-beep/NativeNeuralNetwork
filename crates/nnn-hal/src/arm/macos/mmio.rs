pub unsafe fn mmio_read32(addr: usize) -> u32 {
    let val = core::ptr::read_volatile(addr as *const u32);
    core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
    val
}

pub unsafe fn mmio_write32(addr: usize, value: u32) {
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    core::ptr::write_volatile(addr as *mut u32, value);
}

pub unsafe fn mmio_read64(addr: usize) -> u64 {
    let val = core::ptr::read_volatile(addr as *const u64);
    core::sync::atomic::fence(core::sync::atomic::Ordering::Acquire);
    val
}

pub unsafe fn mmio_write64(addr: usize, value: u64) {
    core::sync::atomic::fence(core::sync::atomic::Ordering::Release);
    core::ptr::write_volatile(addr as *mut u64, value);
}

pub unsafe fn dsb_sy() {
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}

pub unsafe fn isb() {
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

pub unsafe fn dsb_ish() {
    core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
}
