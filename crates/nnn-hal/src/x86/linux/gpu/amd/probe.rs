use crate::x86::linux::syscall;
use super::drm::{iowr, open_render_node, DRM_RADEON_INFO};

const RADEON_INFO_DEVICE_ID: u32 = 0x0000_0000;
const RADEON_INFO_NUM_BACKENDS: u32 = 0x0000_000a;
const RADEON_INFO_NUM_TILE_PIPES: u32 = 0x0000_000b;

#[repr(C)]
struct DrmRadeonInfo {
    request: u32,
    pad: u32,
    value: u64,
}

fn query_u32(fd: i64, request: u32) -> Option<u32> {
    let mut out: u32 = 0;
    let info = DrmRadeonInfo {
        request,
        pad: 0,
        value: (&mut out as *mut u32 as usize) as u64,
    };
    let request_code = iowr(DRM_RADEON_INFO, core::mem::size_of::<DrmRadeonInfo>());
    let rc = syscall::sys_ioctl(fd, request_code, &info as *const DrmRadeonInfo as usize);
    if rc == 0 {
        Some(out)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RadeonProbe {
    pub(crate) device_id: u32,
    pub(crate) num_backends: u32,
    pub(crate) num_tile_pipes: u32,
}

pub(crate) fn probe() -> Option<RadeonProbe> {
    let fd = open_render_node()?;
    let device_id = query_u32(fd, RADEON_INFO_DEVICE_ID);
    let num_backends = query_u32(fd, RADEON_INFO_NUM_BACKENDS).unwrap_or(0);
    let num_tile_pipes = query_u32(fd, RADEON_INFO_NUM_TILE_PIPES).unwrap_or(0);
    let _ = syscall::sys_close(fd);
    device_id.map(|id| RadeonProbe {
        device_id: id,
        num_backends,
        num_tile_pipes,
    })
}
