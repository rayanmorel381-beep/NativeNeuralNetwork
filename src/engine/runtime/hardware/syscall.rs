#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), any(target_os = "linux", target_os = "android")))]
use super::arm::linux::syscall as backend;
#[cfg(all(any(target_arch = "arm", target_arch = "aarch64"), target_os = "macos"))]
use super::arm::macos::syscall as backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), any(target_os = "linux", target_os = "android")))]
use super::x86::linux::syscall as backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "macos"))]
use super::x86::macos::syscall as backend;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_os = "windows"))]
use super::x86::windows::syscall as backend;

pub const O_RDONLY: i32 = backend::O_RDONLY;
pub const O_WRONLY: i32 = backend::O_WRONLY;
pub const O_APPEND: i32 = backend::O_APPEND;

pub fn o_creat() -> i32 { backend::O_CREAT }
pub fn o_trunc() -> i32 { backend::O_TRUNC }

pub fn sys_open(path: &[u8], flags: i32, mode: u32) -> i64 {
    backend::sys_open(path, flags, mode)
}

pub fn sys_close(fd: i64) -> i64 {
    backend::sys_close(fd)
}

pub fn sys_write_fd(fd: i64, buf: &[u8]) -> i64 {
    backend::sys_write_fd(fd, buf)
}

pub fn sys_read_fd(fd: i64, buf: &mut [u8]) -> i64 {
    backend::sys_read_fd(fd, buf)
}

pub fn sys_mkdir(path: &[u8], mode: u32) -> i64 {
    backend::sys_mkdir(path, mode)
}

pub fn mmap_shared_anon(size: usize) -> *mut u8 {
    backend::mmap_shared_anon(size)
}

pub fn munmap(ptr: *mut u8, size: usize) {
    backend::munmap(ptr, size)
}

pub fn monotonic_ns() -> u64 {
    backend::monotonic_ns()
}

pub fn fork() -> i64 {
    backend::fork()
}

pub fn waitpid(pid: i64) {
    backend::waitpid(pid)
}

pub fn exit(code: i32) -> ! {
    backend::exit(code)
}

pub fn set_affinity(mask: usize) {
    backend::set_affinity(mask)
}

#[repr(C)]
pub struct DirEnt64 {
    pub ino: u64,
    pub off: i64,
    pub reclen: u16,
    pub ftype: u8,
    pub name: [u8; 256],
}

pub fn sys_getdents64(fd: i64, buf: &mut [u8]) -> i64 {
    backend::sys_getdents64(fd, buf)
}
