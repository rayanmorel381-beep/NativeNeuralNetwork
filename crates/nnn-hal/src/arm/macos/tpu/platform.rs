use core::sync::atomic::{AtomicUsize, Ordering};

const REG_ID: usize = 0x000;
const REG_CTRL: usize = 0x004;
const REG_STATUS: usize = 0x008;
const REG_CLK_GATE: usize = 0x00C;
const REG_POWER: usize = 0x010;
const REG_COMPUTE_CMD: usize = 0x100;
const REG_COMPUTE_ADDR: usize = 0x104;
const REG_COMPUTE_SIZE: usize = 0x108;
const REG_IRQ_STATUS: usize = 0x020;
const REG_IRQ_ENABLE: usize = 0x024;
const REG_IRQ_CLEAR: usize = 0x028;

const CTRL_RESET: u32 = 1 << 0;
const CTRL_ENABLE: u32 = 1 << 1;
const CLK_COMPUTE: u32 = 1 << 0;
const CLK_MEMORY: u32 = 1 << 1;
const CLK_BUS: u32 = 1 << 2;

const STATUS_READY: u32 = 1 << 0;

const AIC2_MASK_CLR_BASE: usize = 0x4180;
const AIC2_IRQ_CFG_BASE: usize = 0x4000;

static AIC_BASE: AtomicUsize = AtomicUsize::new(0);

pub fn set_aic_base(base: usize) {
    AIC_BASE.store(base, Ordering::Release);
}

pub fn read_device_id(mmio_base: usize) -> u32 {
    unsafe { super::super::mmio::mmio_read32(mmio_base + REG_ID) }
}

pub fn reset_device(mmio_base: usize) {
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_CTRL, CTRL_RESET) };
    let mut timeout = 10_000u32;
    while timeout > 0 {
        unsafe { super::super::mmio::dsb_ish() };
        let status = unsafe { super::super::mmio::mmio_read32(mmio_base + REG_STATUS) };
        if status & STATUS_READY != 0 {
            break;
        }
        timeout -= 1;
    }
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_CTRL, CTRL_ENABLE) };
}

pub fn enable_clocks(mmio_base: usize) {
    unsafe {
        super::super::mmio::mmio_write32(
            mmio_base + REG_CLK_GATE,
            CLK_COMPUTE | CLK_MEMORY | CLK_BUS,
        );
        super::super::mmio::mmio_write32(mmio_base + REG_POWER, 0x01);
        super::super::mmio::dsb_ish();
    }
}

pub fn submit_compute(mmio_base: usize, cmd: u32, data_addr: u32, size: u32) {
    unsafe {
        super::super::mmio::mmio_write32(mmio_base + REG_COMPUTE_ADDR, data_addr);
        super::super::mmio::mmio_write32(mmio_base + REG_COMPUTE_SIZE, size);
        super::super::mmio::mmio_write32(mmio_base + REG_COMPUTE_CMD, cmd);
    }
}

pub fn read_status(mmio_base: usize) -> u32 {
    unsafe { super::super::mmio::mmio_read32(mmio_base + REG_STATUS) }
}

pub fn enable_interrupts(mmio_base: usize) {
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_IRQ_ENABLE, 0x0F) };
}

pub fn clear_interrupts(mmio_base: usize) -> u32 {
    let status = unsafe { super::super::mmio::mmio_read32(mmio_base + REG_IRQ_STATUS) };
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_IRQ_CLEAR, status) };
    status
}

pub fn configure_aic_irq(irq_id: u32, target_cpu: u32) {
    let aic_base = AIC_BASE.load(Ordering::Acquire);
    if aic_base == 0 {
        return;
    }
    let reg_idx = (irq_id / 32) as usize;
    let bit = 1u32 << (irq_id % 32);
    unsafe {
        super::super::mmio::mmio_write32(aic_base + AIC2_MASK_CLR_BASE + reg_idx * 4, bit);
    }
    let cfg_addr = aic_base + AIC2_IRQ_CFG_BASE + irq_id as usize * 4;
    unsafe {
        super::super::mmio::mmio_write32(cfg_addr, target_cpu & 0xFF);
    }
}

pub fn power_on(mmio_base: usize) {
    unsafe {
        super::super::mmio::mmio_write32(mmio_base + REG_POWER, 0x03);
        super::super::mmio::dsb_ish();
    }
}

pub fn power_off(mmio_base: usize) {
    unsafe {
        super::super::mmio::mmio_write32(mmio_base + REG_POWER, 0x00);
        super::super::mmio::dsb_ish();
    }
}
