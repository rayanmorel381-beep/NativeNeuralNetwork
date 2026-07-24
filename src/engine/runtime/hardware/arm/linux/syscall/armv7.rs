use core::arch::asm;

pub const SYS_READ: usize = 3;
pub const SYS_WRITE: usize = 4;
pub const SYS_OPENAT: usize = 322;
pub const SYS_CLOSE: usize = 6;
pub const SYS_MKDIRAT: usize = 323;
pub const SYS_MMAP: usize = 192;
pub const SYS_MUNMAP: usize = 91;
pub const SYS_PRCTL: usize = 172;
pub const SYS_SCHED_SETAFFINITY: usize = 241;
pub const SYS_SCHED_GETAFFINITY: usize = 242;
pub const SYS_SYSINFO: usize = 116;
pub const SYS_CLOCK_GETTIME: usize = 403;
pub const SYS_CLONE: usize = 120;
pub const SYS_WAIT4: usize = 114;
pub const SYS_EXIT: usize = 1;
pub const SYS_GETDENTS64: usize = 217;

#[inline]
pub unsafe fn syscall1(n: usize, a: usize) -> i64 {
    let ret: i32;
    asm!(
        "svc 0",
        in("r7") n,
        inlateout("r0") a => ret,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall2(n: usize, a: usize, b: usize) -> i64 {
    let ret: i32;
    asm!(
        "svc 0",
        in("r7") n,
        inlateout("r0") a => ret,
        in("r1") b,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall3(n: usize, a: usize, b: usize, c: usize) -> i64 {
    let ret: i32;
    asm!(
        "svc 0",
        in("r7") n,
        inlateout("r0") a => ret,
        in("r1") b,
        in("r2") c,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall4(n: usize, a: usize, b: usize, c: usize, d: usize) -> i64 {
    let ret: i32;
    asm!(
        "svc 0",
        in("r7") n,
        inlateout("r0") a => ret,
        in("r1") b,
        in("r2") c,
        in("r3") d,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall5(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize) -> i64 {
    let ret: i32;
    asm!(
        "svc 0",
        in("r7") n,
        inlateout("r0") a => ret,
        in("r1") b,
        in("r2") c,
        in("r3") d,
        in("r4") e,
        options(nostack, preserves_flags),
    );
    ret as i64
}

#[inline]
pub unsafe fn syscall6(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> i64 {
    let ret: i32;
    asm!(
        "svc 0",
        in("r7") n,
        inlateout("r0") a => ret,
        in("r1") b,
        in("r2") c,
        in("r3") d,
        in("r4") e,
        in("r5") f,
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
