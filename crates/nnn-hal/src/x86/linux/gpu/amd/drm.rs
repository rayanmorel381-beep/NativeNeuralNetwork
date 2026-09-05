use crate::x86::linux::syscall;

pub(super) const DRM_COMMAND_BASE: u32 = 0x40;
pub(super) const DRM_RADEON_GEM_CREATE: u32 = 0x1d;
pub(super) const DRM_RADEON_GEM_MMAP: u32 = 0x1e;
pub(super) const DRM_RADEON_GEM_WAIT_IDLE: u32 = 0x24;
pub(super) const DRM_RADEON_GEM_VA: u32 = 0x2b;
pub(super) const DRM_RADEON_CS: u32 = 0x26;
pub(super) const DRM_RADEON_INFO: u32 = 0x27;

pub(super) const RADEON_GEM_DOMAIN_GTT: u32 = 0x2;
pub(super) const RADEON_GEM_DOMAIN_VRAM: u32 = 0x4;

const IOC_WRITE: usize = 1;
const IOC_READ: usize = 2;
const TYPE_DRM: usize = 0x64;

const fn ioc(dir: usize, nr: u32, size: usize) -> usize {
    (dir << 30) | ((size & 0x3fff) << 16) | (TYPE_DRM << 8) | ((DRM_COMMAND_BASE + nr) as usize)
}

const fn ioc_core(dir: usize, nr: u32, size: usize) -> usize {
    (dir << 30) | ((size & 0x3fff) << 16) | (TYPE_DRM << 8) | (nr as usize)
}

pub(super) const fn iowr(nr: u32, size: usize) -> usize {
    ioc(IOC_READ | IOC_WRITE, nr, size)
}

pub(super) const fn iow(nr: u32, size: usize) -> usize {
    ioc(IOC_WRITE, nr, size)
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

fn driver_name_matches(fd: i64, expected: &[u8]) -> bool {
    let mut name_buf = [0u8; 32];
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
    let request = ioc_core(
        IOC_READ | IOC_WRITE,
        0,
        core::mem::size_of::<DrmVersion>(),
    );
    if syscall::sys_ioctl(fd, request, &mut version as *mut DrmVersion as usize) != 0 {
        return false;
    }
    let len = version.name_len as usize;
    len == expected.len() && len <= name_buf.len() && &name_buf[..len] == expected
}

const RENDER_NODES: &[&[u8]] = &[
    b"/dev/dri/renderD128\0",
    b"/dev/dri/renderD129\0",
    b"/dev/dri/card0\0",
];

pub(crate) fn open_render_node() -> Option<i64> {
    for path in RENDER_NODES {
        let fd = syscall::sys_open(path, syscall::O_RDWR, 0);
        if fd >= 0 {
            if driver_name_matches(fd, b"radeon") {
                return Some(fd);
            }
            let _ = syscall::sys_close(fd);
        }
    }
    None
}
