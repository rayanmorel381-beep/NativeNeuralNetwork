use core::sync::atomic::{AtomicBool, Ordering};

const REG_DEVICE_ID: usize = 0x0000;
const REG_REVISION: usize = 0x0004;
const REG_CONTROL: usize = 0x0010;
const REG_STATUS: usize = 0x0014;
const REG_CLOCK_GATE: usize = 0x0020;
const REG_POWER_CTRL: usize = 0x0024;
const REG_TOKEN_INPUT: usize = 0x0100;
const REG_TOKEN_OUTPUT: usize = 0x0104;
const REG_HIDDEN_SIZE: usize = 0x0108;
const REG_BATCH_SIZE: usize = 0x010C;
const REG_QUANT_MODE: usize = 0x0110;
const REG_INFER_CMD: usize = 0x0120;

const CTRL_RESET: u32 = 1 << 0;
const CTRL_ENABLE: u32 = 1 << 1;

const CLK_INFERENCE: u32 = 1 << 0;
const CLK_MEMORY: u32 = 1 << 1;
const CLK_BUS: u32 = 1 << 2;

const PWR_ON: u32 = 1 << 0;

const INFER_PREFILL: u32 = 0x01;
const INFER_DECODE: u32 = 0x02;
const INFER_SPECULATIVE: u32 = 0x04;

const STATUS_READY: u32 = 1 << 0;
const STATUS_BUSY: u32 = 1 << 1;
const STATUS_ERROR: u32 = 1 << 4;

const GIC_DIST_BASE: usize = 0x0800_0000;
const GIC_ISENABLER_OFFSET: usize = 0x0100;
const GIC_ITARGETSR_OFFSET: usize = 0x0800;
const GIC_ICFGR_OFFSET: usize = 0x0C00;

static POWER_ON: AtomicBool = AtomicBool::new(false);

fn mmio_read(base: usize, offset: usize) -> u32 {
    unsafe { super::super::mmio::mmio_read32(base + offset) }
}

fn mmio_write(base: usize, offset: usize, val: u32) {
    unsafe {
        super::super::mmio::mmio_write32(base + offset, val);
    }
}

pub fn read_device_id(base: usize) -> u32 {
    mmio_read(base, REG_DEVICE_ID)
}

pub fn read_revision(base: usize) -> u32 {
    mmio_read(base, REG_REVISION)
}

pub fn reset_device(base: usize) {
    mmio_write(base, REG_CONTROL, CTRL_RESET);
    let mut timeout = 10000u32;
    while timeout > 0 {
        let status = mmio_read(base, REG_STATUS);
        if status & STATUS_READY != 0 {
            break;
        }
        timeout -= 1;
    }
    mmio_write(base, REG_CONTROL, CTRL_ENABLE);
}

pub fn enable_clocks(base: usize) {
    mmio_write(base, REG_CLOCK_GATE, CLK_INFERENCE | CLK_MEMORY | CLK_BUS);
}

pub fn power_on(base: usize) {
    mmio_write(base, REG_POWER_CTRL, PWR_ON);
    let mut timeout = 5000u32;
    while timeout > 0 {
        let status = mmio_read(base, REG_STATUS);
        if status & STATUS_READY != 0 {
            break;
        }
        timeout -= 1;
    }
    POWER_ON.store(true, Ordering::Release);
}

pub fn power_off(base: usize) {
    mmio_write(base, REG_POWER_CTRL, 0);
    POWER_ON.store(false, Ordering::Release);
}

pub fn is_powered() -> bool {
    POWER_ON.load(Ordering::Acquire)
}

pub fn configure_inference(base: usize, hidden_size: u32, batch_size: u32, quant_mode: u32) {
    mmio_write(base, REG_HIDDEN_SIZE, hidden_size);
    mmio_write(base, REG_BATCH_SIZE, batch_size);
    mmio_write(base, REG_QUANT_MODE, quant_mode);
}

pub fn submit_prefill(base: usize, token_addr: u32, token_count: u32) {
    mmio_write(base, REG_TOKEN_INPUT, token_addr);
    mmio_write(base, REG_TOKEN_OUTPUT, token_count);
    mmio_write(base, REG_INFER_CMD, INFER_PREFILL);
}

pub fn submit_decode(base: usize, token_addr: u32) {
    mmio_write(base, REG_TOKEN_INPUT, token_addr);
    mmio_write(base, REG_INFER_CMD, INFER_DECODE);
}

pub fn submit_speculative(base: usize, token_addr: u32, draft_count: u32) {
    mmio_write(base, REG_TOKEN_INPUT, token_addr);
    mmio_write(base, REG_TOKEN_OUTPUT, draft_count);
    mmio_write(base, REG_INFER_CMD, INFER_SPECULATIVE);
}

pub fn is_ready(base: usize) -> bool {
    mmio_read(base, REG_STATUS) & STATUS_READY != 0
}

pub fn is_busy(base: usize) -> bool {
    mmio_read(base, REG_STATUS) & STATUS_BUSY != 0
}

pub fn has_error(base: usize) -> bool {
    mmio_read(base, REG_STATUS) & STATUS_ERROR != 0
}

pub fn configure_gic_spi(spi_id: u32, cpu_target: u8) {
    let irq = spi_id + 32;
    let reg_index = (irq / 32) as usize;
    let bit_offset = irq % 32;

    let enable_addr = GIC_DIST_BASE + GIC_ISENABLER_OFFSET + reg_index * 4;
    unsafe {
        super::super::mmio::mmio_write32(enable_addr, 1 << bit_offset);
    }

    let target_reg = (irq / 4) as usize;
    let target_shift = (irq % 4) * 8;
    let target_addr = GIC_DIST_BASE + GIC_ITARGETSR_OFFSET + target_reg * 4;
    unsafe {
        let current = super::super::mmio::mmio_read32(target_addr);
        let mask = !(0xFF << target_shift);
        let val = (current & mask) | ((cpu_target as u32) << target_shift);
        super::super::mmio::mmio_write32(target_addr, val);
    }

    let cfg_reg = (irq / 16) as usize;
    let cfg_shift = ((irq % 16) * 2) + 1;
    let cfg_addr = GIC_DIST_BASE + GIC_ICFGR_OFFSET + cfg_reg * 4;
    unsafe {
        let current = super::super::mmio::mmio_read32(cfg_addr);
        let val = current | (1 << cfg_shift);
        super::super::mmio::mmio_write32(cfg_addr, val);
    }
}
