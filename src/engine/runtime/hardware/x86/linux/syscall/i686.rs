use core::arch::asm;

pub const SYS_READ: usize = 3;
pub const SYS_WRITE: usize = 4;
pub const SYS_OPEN: usize = 5;
pub const SYS_CLOSE: usize = 6;
pub const SYS_MKDIR: usize = 39;
pub const SYS_OLDMMAP: usize = 90;
pub const SYS_MUNMAP: usize = 91;
pub const SYS_SCHED_SETAFFINITY: usize = 241;
pub const SYS_SCHED_GETAFFINITY: usize = 242;
pub const SYS_SYSINFO: usize = 116;
pub const SYS_CLOCK_GETTIME: usize = 403;
pub const SYS_FORK: usize = 2;
pub const SYS_WAIT4: usize = 114;
pub const SYS_EXIT: usize = 1;
pub const SYS_PREAD64: usize = 180;
pub const SYS_IOCTL: usize = 54;
pub const SYS_GETDENTS64: usize = 220;

#[inline]
pub unsafe fn syscall1(n: usize, a: usize) -> i64 {
    let ret: i32;
    asm!(
        "push ebx",
        "mov ebx, {a}",
        "int 0x80",
        "pop ebx",
        a = in(reg) a,
        inlateout("eax") n => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall2(n: usize, a: usize, b: usize) -> i64 {
    let ret: i32;
    asm!(
        "push ebx",
        "mov ebx, {a}",
        "int 0x80",
        "pop ebx",
        a = in(reg) a,
        in("ecx") b,
        inlateout("eax") n => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall3(n: usize, a: usize, b: usize, c: usize) -> i64 {
    let ret: i32;
    asm!(
        "push ebx",
        "mov ebx, {a}",
        "int 0x80",
        "pop ebx",
        a = in(reg) a,
        in("ecx") b,
        in("edx") c,
        inlateout("eax") n => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall4(n: usize, a: usize, b: usize, c: usize, d: usize) -> i64 {
    let ret: i32;
    asm!(
        "push ebx",
        "push esi",
        "mov ebx, {a}",
        "mov esi, {d}",
        "int 0x80",
        "pop esi",
        "pop ebx",
        a = in(reg) a,
        d = in(reg) d,
        in("ecx") b,
        in("edx") c,
        inlateout("eax") n => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall5(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> i64 {
    let ret: i32;
    asm!(
        "push ebx",
        "push esi",
        "mov ebx, {a}",
        "mov esi, {d}",
        "int 0x80",
        "pop esi",
        "pop ebx",
        a = in(reg) a,
        d = in(reg) d,
        in("ecx") b,
        in("edx") c,
        in("edi") e,
        inlateout("eax") n => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn sys_pread64_impl(fd: usize, buf: usize, count: usize, off_lo: usize, off_hi: usize) -> i64 {
    syscall5(SYS_PREAD64, fd, buf, count, off_lo, off_hi)
}

#[inline]
pub unsafe fn sys_open_impl(path: usize, flags: usize, mode: usize) -> i64 {
    syscall3(SYS_OPEN, path, flags, mode)
}

#[inline]
pub unsafe fn sys_ioctl_impl(fd: usize, request: usize, arg: usize) -> i64 {
    syscall3(SYS_IOCTL, fd, request, arg)
}

#[inline]
pub unsafe fn sys_mkdir_impl(path: usize, mode: usize) -> i64 {
    syscall2(SYS_MKDIR, path, mode)
}

#[repr(C)]
struct OldMmapArgs {
    addr: usize,
    len: usize,
    prot: usize,
    flags: usize,
    fd: usize,
    offset: usize,
}

#[inline]
pub unsafe fn sys_mmap_anon(size: usize, prot: usize, flags: usize) -> i64 {
    let args = OldMmapArgs {
        addr: 0,
        len: size,
        prot,
        flags,
        fd: (-1isize) as usize,
        offset: 0,
    };
    syscall1(SYS_OLDMMAP, &args as *const OldMmapArgs as usize)
}

#[inline]
pub unsafe fn sys_mmap_fd_impl(size: usize, prot: usize, flags: usize, fd: usize, offset: usize) -> i64 {
    let args = OldMmapArgs {
        addr: 0,
        len: size,
        prot,
        flags,
        fd,
        offset,
    };
    syscall1(SYS_OLDMMAP, &args as *const OldMmapArgs as usize)
}

#[inline]
pub unsafe fn sys_fork_impl() -> i64 {
    syscall1(SYS_FORK, 0)
}
