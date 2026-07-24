#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "arm")]
mod armv7;

#[cfg(target_arch = "aarch64")]
use aarch64 as arch;
#[cfg(target_arch = "arm")]
use armv7 as arch;

pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_CREAT: i32 = 0o100;
pub const O_TRUNC: i32 = 0o1000;
pub const O_APPEND: i32 = 0o2000;

const CLOCK_MONOTONIC: i32 = 1;
const PROT_READ_WRITE: i32 = 0x3;
const MAP_SHARED_ANON: i32 = 0x21;
const AT_FDCWD: i32 = -100;

#[repr(C)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

pub fn sys_open(path: &[u8], flags: i32, mode: u32) -> i64 {
    unsafe {
        arch::syscall4(
            arch::SYS_OPENAT,
            AT_FDCWD as usize,
            path.as_ptr() as usize,
            flags as usize,
            mode as usize,
        )
    }
}

pub fn sys_close(fd: i64) -> i64 {
    unsafe { arch::syscall1(arch::SYS_CLOSE, fd as usize) }
}

pub fn sys_write_fd(fd: i64, buf: &[u8]) -> i64 {
    unsafe {
        arch::syscall3(
            arch::SYS_WRITE,
            fd as usize,
            buf.as_ptr() as usize,
            buf.len(),
        )
    }
}

pub fn sys_read_fd(fd: i64, buf: &mut [u8]) -> i64 {
    unsafe {
        arch::syscall3(
            arch::SYS_READ,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
        )
    }
}

pub fn sys_mkdir(path: &[u8], mode: u32) -> i64 {
    unsafe {
        arch::syscall3(
            arch::SYS_MKDIRAT,
            AT_FDCWD as usize,
            path.as_ptr() as usize,
            mode as usize,
        )
    }
}

pub fn mmap_shared_anon(size: usize) -> *mut u8 {
    let ret = unsafe {
        arch::sys_mmap_anon(size, PROT_READ_WRITE as usize, MAP_SHARED_ANON as usize)
    };
    if ret < 0 { core::ptr::null_mut() } else { ret as *mut u8 }
}

pub fn munmap(ptr: *mut u8, size: usize) {
    unsafe { arch::syscall2(arch::SYS_MUNMAP, ptr as usize, size); }
}

pub fn monotonic_ns() -> u64 {
    let mut ts = Timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe {
        arch::syscall2(
            arch::SYS_CLOCK_GETTIME,
            CLOCK_MONOTONIC as usize,
            &mut ts as *mut Timespec as usize,
        );
    }
    (ts.tv_sec as u64).saturating_mul(1_000_000_000).saturating_add(ts.tv_nsec as u64)
}

pub fn fork() -> i64 {
    unsafe { arch::sys_fork_impl() }
}

pub fn waitpid(pid: i64) {
    unsafe { arch::syscall4(arch::SYS_WAIT4, pid as usize, 0, 0, 0); }
}

pub fn exit(code: i32) -> ! {
    unsafe {
        arch::syscall1(arch::SYS_EXIT, code as usize);
        core::hint::unreachable_unchecked()
    }
}

pub fn set_affinity(mask: usize) {
    unsafe {
        arch::syscall3(
            arch::SYS_SCHED_SETAFFINITY,
            0,
            core::mem::size_of::<usize>(),
            &mask as *const usize as usize,
        );
    }
}

#[repr(C)]
pub struct SysInfo {
    pub uptime: isize,
    pub loads: [usize; 3],
    pub totalram: usize,
    pub freeram: usize,
    pub sharedram: usize,
    pub bufferram: usize,
    pub totalswap: usize,
    pub freeswap: usize,
    pub procs: u16,
    pub _pad1: u16,
    pub totalhigh: usize,
    pub freehigh: usize,
    pub mem_unit: u32,
    pub _pad2: u32,
}

pub fn sys_sched_getaffinity(pid: i64, mask: &mut [usize]) -> i64 {
    let len = core::mem::size_of_val(mask);
    unsafe {
        arch::syscall3(
            arch::SYS_SCHED_GETAFFINITY,
            pid as usize,
            len,
            mask.as_mut_ptr() as usize,
        )
    }
}

pub fn sys_sysinfo(info: &mut SysInfo) -> i64 {
    unsafe {
        arch::syscall1(
            arch::SYS_SYSINFO,
            info as *mut SysInfo as usize,
        )
    }
}

#[repr(C, packed)]
pub struct DirEnt64 {
    pub ino: u64,
    pub off: i64,
    pub reclen: u16,
    pub ftype: u8,
}

pub fn sys_getdents64(fd: i64, buf: &mut [u8]) -> i64 {
    unsafe {
        arch::syscall3(
            arch::SYS_GETDENTS64,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
        )
    }
}

const FUTEX_WAIT: usize = 0;
const FUTEX_WAKE: usize = 1;

pub fn futex_wait_u32(addr: *const u32, val: u32) {
    unsafe {
        arch::syscall4(
            arch::SYS_FUTEX,
            addr as usize,
            FUTEX_WAIT,
            val as usize,
            0,
        );
    }
}

pub fn futex_wake_u32(addr: *const u32, n: u32) {
    unsafe {
        arch::syscall3(
            arch::SYS_FUTEX,
            addr as usize,
            FUTEX_WAKE,
            n as usize,
        );
    }
}

pub fn prctl_set_pdeathsig(sig: usize) {
    const PR_SET_PDEATHSIG: usize = 1;
    unsafe { arch::syscall2(arch::SYS_PRCTL, PR_SET_PDEATHSIG, sig); }
}
