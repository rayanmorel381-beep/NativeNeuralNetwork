use crate::engine::runtime::hardware::x86::linux::syscall;
use super::drm::{
    iow, iowr, DRM_RADEON_GEM_CREATE, DRM_RADEON_GEM_MMAP, DRM_RADEON_GEM_VA,
    DRM_RADEON_GEM_WAIT_IDLE,
};

const RADEON_VA_MAP: u32 = 1;
const RADEON_VM_PAGE_READABLE: u32 = 2;
const RADEON_VM_PAGE_WRITEABLE: u32 = 4;
const RADEON_VM_PAGE_SNOOPED: u32 = 16;
const RADEON_VA_RESULT_OK: u32 = 0;
const RADEON_VA_RESULT_VA_EXIST: u32 = 2;

#[repr(C)]
struct GemCreate {
    size: u64,
    alignment: u64,
    handle: u32,
    initial_domain: u32,
    flags: u32,
}

#[repr(C)]
struct GemMmap {
    handle: u32,
    pad: u32,
    offset: u64,
    size: u64,
    addr_ptr: u64,
}

#[repr(C)]
struct GemWaitIdle {
    handle: u32,
    pad: u32,
}

#[repr(C)]
struct GemVa {
    handle: u32,
    operation: u32,
    vm_id: u32,
    flags: u32,
    offset: u64,
}

pub(crate) struct GemBuffer {
    pub(crate) handle: u32,
    pub(crate) size: usize,
    pub(crate) cpu_ptr: *mut u8,
    pub(crate) va: u64,
    pub(crate) domain: u32,
    fd: i64,
}

pub(crate) fn gem_create(fd: i64, size: usize, domain: u32, flags: u32) -> Option<u32> {
    let mut args = GemCreate {
        size: size as u64,
        alignment: 4096,
        handle: 0,
        initial_domain: domain,
        flags,
    };
    let code = iowr(DRM_RADEON_GEM_CREATE, core::mem::size_of::<GemCreate>());
    let rc = syscall::sys_ioctl(fd, code, &mut args as *mut GemCreate as usize);
    if rc != 0 {
        return None;
    }
    Some(args.handle)
}

fn gem_mmap_offset(fd: i64, handle: u32, size: usize) -> Option<u64> {
    let mut args = GemMmap {
        handle,
        pad: 0,
        offset: 0,
        size: size as u64,
        addr_ptr: 0,
    };
    let code = iowr(DRM_RADEON_GEM_MMAP, core::mem::size_of::<GemMmap>());
    let rc = syscall::sys_ioctl(fd, code, &mut args as *mut GemMmap as usize);
    if rc != 0 {
        return None;
    }
    Some(args.addr_ptr)
}

pub(crate) fn alloc_gtt(fd: i64, size: usize) -> Option<GemBuffer> {
    let aligned = (size + 4095) & !4095;
    let handle = gem_create(fd, aligned, super::drm::RADEON_GEM_DOMAIN_GTT, 0)?;
    let offset = gem_mmap_offset(fd, handle, aligned)?;
    let ptr = syscall::mmap_shared_fd(fd, aligned, offset);
    if ptr.is_null() {
        return None;
    }
    Some(GemBuffer {
        handle,
        size: aligned,
        cpu_ptr: ptr,
        va: 0,
        domain: super::drm::RADEON_GEM_DOMAIN_GTT,
        fd,
    })
}

pub(crate) fn alloc_vram(fd: i64, size: usize) -> Option<GemBuffer> {
    let aligned = (size + 4095) & !4095;
    let handle = gem_create(fd, aligned, super::drm::RADEON_GEM_DOMAIN_VRAM, 0)?;
    let offset = gem_mmap_offset(fd, handle, aligned)?;
    let ptr = syscall::mmap_shared_fd(fd, aligned, offset);
    if ptr.is_null() {
        return None;
    }
    Some(GemBuffer {
        handle,
        size: aligned,
        cpu_ptr: ptr,
        va: 0,
        domain: super::drm::RADEON_GEM_DOMAIN_VRAM,
        fd,
    })
}

fn gem_va_map(fd: i64, handle: u32, va: u64) -> bool {
    let mut args = GemVa {
        handle,
        operation: RADEON_VA_MAP,
        vm_id: 0,
        flags: RADEON_VM_PAGE_READABLE | RADEON_VM_PAGE_WRITEABLE | RADEON_VM_PAGE_SNOOPED,
        offset: va,
    };
    let code = iowr(DRM_RADEON_GEM_VA, core::mem::size_of::<GemVa>());
    let rc = syscall::sys_ioctl(fd, code, &mut args as *mut GemVa as usize);
    rc == 0 && (args.operation == RADEON_VA_RESULT_OK || args.operation == RADEON_VA_RESULT_VA_EXIST)
}

impl GemBuffer {
    pub(crate) fn bind_va(&mut self, va: u64) -> bool {
        if gem_va_map(self.fd, self.handle, va) {
            self.va = va;
            true
        } else {
            false
        }
    }

    pub(crate) fn wait_idle(&self) -> bool {
        let mut args = GemWaitIdle {
            handle: self.handle,
            pad: 0,
        };
        let code = iow(DRM_RADEON_GEM_WAIT_IDLE, core::mem::size_of::<GemWaitIdle>());
        syscall::sys_ioctl(self.fd, code, &mut args as *mut GemWaitIdle as usize) == 0
    }
}

impl Drop for GemBuffer {
    fn drop(&mut self) {
        if !self.cpu_ptr.is_null() {
            syscall::munmap(self.cpu_ptr, self.size);
        }
    }
}
