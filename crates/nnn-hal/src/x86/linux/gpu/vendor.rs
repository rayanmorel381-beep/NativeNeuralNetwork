use super::{amd, apple, intel};
use crate::x86::linux::syscall;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Vendor {
    Amd,
    Intel,
    Apple,
}

#[repr(C)]
struct DrmVersion {
    version_major: i32,
    version_minor: i32,
    version_patchlevel: i32,
    name_len: u64,
    name: u64,
    date_len: u64,
    date: u64,
    desc_len: u64,
    desc: u64,
}

const DRM_IOCTL_TYPE: usize = 0x64;
const DRM_IOC_READ_WRITE: usize = 3;
const DRM_VERSION_NR: usize = 0x00;

const DRM_NODES: &[&[u8]] = &[
    b"/dev/dri/renderD128\0",
    b"/dev/dri/renderD129\0",
    b"/dev/dri/card0\0",
];

fn read_drm_driver_name(fd: i64, name_buf: &mut [u8]) -> usize {
    let mut version = DrmVersion {
        version_major: 0,
        version_minor: 0,
        version_patchlevel: 0,
        name_len: name_buf.len() as u64,
        name: name_buf.as_mut_ptr() as u64,
        date_len: 0,
        date: 0,
        desc_len: 0,
        desc: 0,
    };
    let request = (DRM_IOC_READ_WRITE << 30)
        | ((core::mem::size_of::<DrmVersion>() & 0x3fff) << 16)
        | (DRM_IOCTL_TYPE << 8)
        | DRM_VERSION_NR;
    if syscall::sys_ioctl(fd, request, &mut version as *mut DrmVersion as usize) != 0 {
        return 0;
    }
    let len = version.name_len as usize;
    if len <= name_buf.len() {
        len
    } else {
        0
    }
}

fn detect_gpu_vendor_from_drm() -> Option<Vendor> {
    for path in DRM_NODES {
        let fd = syscall::sys_open(path, syscall::O_RDWR, 0);
        if fd < 0 {
            continue;
        }
        let mut name = [0u8; 32];
        let len = read_drm_driver_name(fd, &mut name);
        let _ = syscall::sys_close(fd);
        if len == 0 {
            continue;
        }
        let driver = &name[..len];
        if driver.starts_with(b"radeon") || driver.starts_with(b"amdgpu") {
            return Some(Vendor::Amd);
        }
        if driver.starts_with(b"i915") || driver.starts_with(b"xe") {
            return Some(Vendor::Intel);
        }
    }
    None
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuConfig {
    pub(crate) vendor: Vendor,
    pub(crate) workgroup_size: usize,
    pub(crate) compute_queues: usize,
    pub(crate) render_threads: usize,
    pub(crate) double_buffered: bool,
    pub(crate) frame_budget_us: u64,
    pub(crate) low_power: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GpuSchedule {
    pub(crate) chunks: usize,
    pub(crate) chunk_size: usize,
    pub(crate) frame_budget_us: u64,
}

fn detect_vendor() -> Vendor {
    if let Some(vendor) = detect_gpu_vendor_from_drm() {
        return vendor;
    }
    if let Some(id) = crate::x86::process_identifier() {
        if id.contains("apple") {
            return Vendor::Apple;
        }
        if id.contains("amd") {
            return Vendor::Amd;
        }
        if id.contains("intel") {
            return Vendor::Intel;
        }
    }
    Vendor::Apple
}

pub(crate) fn default_config() -> GpuConfig {
    let vendor = detect_vendor();
    let (workgroup_size, compute_queues, render_threads, double_buffered, frame_budget_us, low_power) = match vendor {
        Vendor::Amd => {
            let c = amd::backend::default_backend_config();
            (c.workgroup_size, c.compute_queues, c.render_threads, c.double_buffered, c.frame_budget_us, c.low_power)
        }
        Vendor::Intel => {
            let c = intel::backend::default_backend_config();
            (c.workgroup_size, c.compute_queues, c.render_threads, c.double_buffered, c.frame_budget_us, c.low_power)
        }
        Vendor::Apple => {
            let c = apple::backend::default_backend_config();
            (c.workgroup_size, c.compute_queues, c.render_threads, c.double_buffered, c.frame_budget_us, c.low_power)
        }
    };
    GpuConfig { vendor, workgroup_size, compute_queues, render_threads, double_buffered, frame_budget_us, low_power }
}

pub(crate) fn clamp_workers(requested: usize) -> usize {
    match detect_vendor() {
        Vendor::Amd   => amd::backend::clamp_workers(requested),
        Vendor::Intel => intel::backend::clamp_workers(requested),
        Vendor::Apple => apple::backend::clamp_workers(requested),
    }
}

pub(crate) fn build_schedule(work_items: usize) -> GpuSchedule {
    let vendor = detect_vendor();
    let (chunks, chunk_size, frame_budget_us) = match vendor {
        Vendor::Amd => {
            let s = amd::scheduler::build_schedule(work_items);
            (s.chunks, s.chunk_size, s.frame_budget_us)
        }
        Vendor::Intel => {
            let s = intel::scheduler::build_schedule(work_items);
            (s.chunks, s.chunk_size, s.frame_budget_us)
        }
        Vendor::Apple => {
            let s = apple::scheduler::build_schedule(work_items);
            (s.chunks, s.chunk_size, s.frame_budget_us)
        }
    };
    GpuSchedule { chunks, chunk_size, frame_budget_us }
}
