use core::arch::asm;

pub const SYS_READ: usize = 63;
pub const SYS_WRITE: usize = 64;
pub const SYS_OPENAT: usize = 56;
pub const SYS_CLOSE: usize = 57;
pub const SYS_MKDIRAT: usize = 34;
pub const SYS_MMAP: usize = 222;
pub const SYS_MUNMAP: usize = 215;
pub const SYS_FUTEX: usize = 98;
pub const SYS_PRCTL: usize = 167;
pub const SYS_SCHED_SETAFFINITY: usize = 122;
pub const SYS_SCHED_GETAFFINITY: usize = 123;
pub const SYS_SYSINFO: usize = 179;
pub const SYS_CLOCK_GETTIME: usize = 113;
pub const SYS_CLONE: usize = 220;
pub const SYS_WAIT4: usize = 260;
pub const SYS_EXIT: usize = 93;
pub const SYS_GETDENTS64: usize = 61;

#[inline]
pub unsafe fn syscall1(n: usize, a: usize) -> i64 {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall2(n: usize, a: usize, b: usize) -> i64 {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a => ret,
        in("x1") b,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall3(n: usize, a: usize, b: usize, c: usize) -> i64 {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a => ret,
        in("x1") b,
        in("x2") c,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall4(n: usize, a: usize, b: usize, c: usize, d: usize) -> i64 {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a => ret,
        in("x1") b,
        in("x2") c,
        in("x3") d,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall5(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> i64 {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a => ret,
        in("x1") b,
        in("x2") c,
        in("x3") d,
        in("x4") e,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall6(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> i64 {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a => ret,
        in("x1") b,
        in("x2") c,
        in("x3") d,
        in("x4") e,
        in("x5") f,
        options(nostack, preserves_flags),
    );
    ret as i64
}

const SIGCHLD: usize = 17;

#[inline]
pub unsafe fn sys_fork_impl() -> i64 {
    syscall5(SYS_CLONE, SIGCHLD, 0, 0, 0, 0)
}

#[inline]
pub unsafe fn sys_mmap_anon(size: usize, prot: usize, flags: usize) -> i64 {
    syscall6(SYS_MMAP, 0, size, prot, flags, (-1i64) as usize, 0)
}
