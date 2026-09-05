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

const GIC_DIST_BASE: usize = 0x0800_0000;
const GIC_ISENABLER: usize = 0x100;
const GIC_ITARGETSR: usize = 0x800;

pub fn read_device_id(mmio_base: usize) -> u32 {
    unsafe { super::super::mmio::mmio_read32(mmio_base + REG_ID) }
}

pub fn reset_device(mmio_base: usize) {
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_CTRL, CTRL_RESET) };
    let mut timeout = 10000u32;
    while timeout > 0 {
        let status = unsafe { super::super::mmio::mmio_read32(mmio_base + REG_STATUS) };
        if status & 0x01 != 0 {
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

pub fn configure_gic_spi(spi_id: u32, target_cpu: u32) {
    let reg_index = (spi_id / 32) as usize;
    let bit_index = spi_id % 32;
    unsafe {
        super::super::mmio::mmio_write32(
            GIC_DIST_BASE + GIC_ISENABLER + reg_index * 4,
            1u32 << bit_index,
        );
    }
    let target_index = spi_id as usize;
    unsafe {
        super::super::mmio::mmio_write32(
            GIC_DIST_BASE + GIC_ITARGETSR + target_index * 4,
            1u32 << target_cpu,
        );
    }
}

pub fn power_on(mmio_base: usize) {
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_POWER, 0x03) };
}

pub fn power_off(mmio_base: usize) {
    unsafe { super::super::mmio::mmio_write32(mmio_base + REG_POWER, 0x00) };
}
