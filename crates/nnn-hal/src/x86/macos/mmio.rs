pub unsafe fn mmio_read32(addr: usize) -> u32 {
    core::ptr::read_volatile(addr as *const u32)
}

pub unsafe fn mmio_write32(addr: usize, value: u32) {
    core::ptr::write_volatile(addr as *mut u32, value)
}
