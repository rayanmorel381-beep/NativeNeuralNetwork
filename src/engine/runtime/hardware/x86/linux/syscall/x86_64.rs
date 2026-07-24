use core::arch::asm;

pub const SYS_READ: usize = 0;
pub const SYS_WRITE: usize = 1;
pub const SYS_OPEN: usize = 2;
pub const SYS_CLOSE: usize = 3;
pub const SYS_MMAP: usize = 9;
pub const SYS_MUNMAP: usize = 11;
pub const SYS_IOCTL: usize = 16;
pub const SYS_MKDIR: usize = 83;
pub const SYS_FUTEX: usize = 202;
pub const SYS_PRCTL: usize = 157;
pub const SYS_SCHED_SETAFFINITY: usize = 203;
pub const SYS_SCHED_GETAFFINITY: usize = 204;
pub const SYS_SYSINFO: usize = 99;
pub const SYS_CLOCK_GETTIME: usize = 228;
pub const SYS_FORK: usize = 57;
pub const SYS_CLONE: usize = 56;
pub const SYS_WAIT4: usize = 61;
pub const SYS_EXIT: usize = 60;
pub const SYS_PREAD64: usize = 17;
pub const SYS_GETDENTS64: usize = 217;

#[inline]
pub unsafe fn syscall1(n: usize, a: usize) -> i64 {
    let ret: i64;
    asm!(
        "syscall",
        in("rax") n,
        inlateout("rdi") a => _,
        lateout("rcx") _,
        lateout("r11") _,
        lateout("rax") ret,
        options(nostack, preserves_flags),
    );
    ret
}

#[inline]
pub unsafe fn syscall2(n: usize, a: usize, b: usize) -> i64 {
    let ret: i64;
    asm!(
        "syscall",
        in("rax") n,
        in("rdi") a,
        in("rsi") b,
        lateout("rcx") _,
        lateout("r11") _,
        lateout("rax") ret,
        options(nostack, preserves_flags),
    );
    ret
}

#[inline]
pub unsafe fn syscall3(n: usize, a: usize, b: usize, c: usize) -> i64 {
    let ret: i64;
    asm!(
        "syscall",
        in("rax") n,
        in("rdi") a,
        in("rsi") b,
        in("rdx") c,
        lateout("rcx") _,
        lateout("r11") _,
        lateout("rax") ret,
        options(nostack, preserves_flags),
    );
    ret
}

#[inline]
pub unsafe fn syscall4(n: usize, a: usize, b: usize, c: usize, d: usize) -> i64 {
    let ret: i64;
    asm!(
        "syscall",
        in("rax") n,
        in("rdi") a,
        in("rsi") b,
        in("rdx") c,
        in("r10") d,
        lateout("rcx") _,
        lateout("r11") _,
        lateout("rax") ret,
        options(nostack, preserves_flags),
    );
    ret
}

#[inline]
pub unsafe fn syscall6(n: usize, a: usize, b: usize, c: usize, d: usize, e: usize, f: usize) -> i64 {
    let ret: i64;
    asm!(
        "syscall",
        in("rax") n,
        in("rdi") a,
        in("rsi") b,
        in("rdx") c,
        in("r10") d,
        in("r8") e,
        in("r9") f,
        lateout("rcx") _,
        lateout("r11") _,
        lateout("rax") ret,
        options(nostack, preserves_flags),
    );
    ret
}

#[inline]
pub unsafe fn sys_open_impl(path: usize, flags: usize, mode: usize) -> i64 {
    syscall3(SYS_OPEN, path, flags, mode)
}

#[inline]
pub unsafe fn sys_mkdir_impl(path: usize, mode: usize) -> i64 {
    syscall2(SYS_MKDIR, path, mode)
}

#[inline]
pub unsafe fn sys_mmap_anon(size: usize, prot: usize, flags: usize) -> i64 {
    syscall6(SYS_MMAP, 0, size, prot, flags, (-1i64) as usize, 0)
}

#[inline]
pub unsafe fn sys_mmap_fd_impl(size: usize, prot: usize, flags: usize, fd: usize, offset: usize) -> i64 {
    syscall6(SYS_MMAP, 0, size, prot, flags, fd, offset)
}

#[inline]
pub unsafe fn sys_fork_impl() -> i64 {
    syscall1(SYS_FORK, 0)
}

#[inline]
pub unsafe fn sys_pread64_impl(fd: usize, buf: usize, count: usize, offset: usize) -> i64 {
    syscall4(SYS_PREAD64, fd, buf, count, offset)
}

#[inline]
pub unsafe fn sys_ioctl_impl(fd: usize, request: usize, arg: usize) -> i64 {
    syscall3(SYS_IOCTL, fd, request, arg)
}

#[inline(never)]
pub unsafe fn sys_clone_thread_impl(
    flags: usize,
    child_stack_top: usize,
    entry: extern "C" fn(usize),
    arg: usize,
    done_ptr: usize,
) -> i64 {
    let ret: i64;
    let entry_addr = entry as usize;
    asm!(
        "syscall",
        "test rax, rax",
        "jnz 2f",
        "mov rdi, r13",
        "call r12",
        "lock inc dword ptr [r14]",
        "mov rax, 202",
        "mov rdi, r14",
        "mov rsi, 1",
        "mov rdx, 1",
        "xor r10, r10",
        "syscall",
        "mov rax, 60",
        "xor rdi, rdi",
        "syscall",
        "2:",
        inlateout("rax") SYS_CLONE => ret,
        inlateout("rdi") flags => _,
        inlateout("rsi") child_stack_top => _,
        inlateout("rdx") 0usize => _,
        inlateout("r10") 0usize => _,
        inlateout("r8") 0usize => _,
        in("r12") entry_addr,
        in("r13") arg,
        in("r14") done_ptr,
        lateout("rcx") _,
        lateout("r11") _,
    );
    ret
}
