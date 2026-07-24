pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_CREAT: i32 = 0x40;
pub const O_TRUNC: i32 = 0x200;
pub const O_APPEND: i32 = 0x8;

const GENERIC_READ: u32 = 0x80000000;
const GENERIC_WRITE: u32 = 0x40000000;
const FILE_SHARE_READ: u32 = 0x1;
const FILE_SHARE_WRITE: u32 = 0x2;
const CREATE_ALWAYS: u32 = 2;
const OPEN_EXISTING: u32 = 3;
const OPEN_ALWAYS: u32 = 4;
const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
const FILE_END: u32 = 2;
const INVALID_HANDLE_VALUE: isize = -1;

const MEM_COMMIT: u32 = 0x1000;
const MEM_RESERVE: u32 = 0x2000;
const MEM_RELEASE: u32 = 0x8000;
const PAGE_READWRITE: u32 = 0x4;

mod ffi {
    use core::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CreateFileA(
            name: *const u8,
            access: u32,
            share: u32,
            sec: *mut c_void,
            disposition: u32,
            attrs: u32,
            template: isize,
        ) -> isize;
        pub fn CloseHandle(h: isize) -> i32;
        pub fn ReadFile(h: isize, buf: *mut c_void, n: u32, read: *mut u32, ov: *mut c_void) -> i32;
        pub fn WriteFile(h: isize, buf: *const c_void, n: u32, written: *mut u32, ov: *mut c_void) -> i32;
        pub fn SetFilePointer(h: isize, dist: i32, high: *mut i32, method: u32) -> u32;
        pub fn CreateDirectoryA(path: *const u8, sec: *mut c_void) -> i32;
        pub fn VirtualAlloc(addr: *mut c_void, size: usize, alloc_type: u32, protect: u32) -> *mut c_void;
        pub fn VirtualFree(addr: *mut c_void, size: usize, free_type: u32) -> i32;
        pub fn QueryPerformanceCounter(c: *mut i64) -> i32;
        pub fn QueryPerformanceFrequency(f: *mut i64) -> i32;
        pub fn ExitProcess(code: u32) -> !;
        pub fn GetCurrentThread() -> isize;
        pub fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
    }
}

pub fn sys_open(path: &[u8], flags: i32, mode: u32) -> i64 {
    let _ = mode;
    let read = flags == O_RDONLY;
    let write = (flags & O_WRONLY) != 0 || (flags & O_APPEND) != 0;
    let create = (flags & O_CREAT) != 0;
    let trunc = (flags & O_TRUNC) != 0;
    let append = (flags & O_APPEND) != 0;
    let access = if read && !write { GENERIC_READ }
        else if write && !read { GENERIC_WRITE }
        else { GENERIC_READ | GENERIC_WRITE };
    let disposition = if create && trunc { CREATE_ALWAYS }
        else if create { OPEN_ALWAYS }
        else { OPEN_EXISTING };
    let h = unsafe {
        ffi::CreateFileA(
            path.as_ptr(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            core::ptr::null_mut(),
            disposition,
            FILE_ATTRIBUTE_NORMAL,
            0,
        )
    };
    if h == INVALID_HANDLE_VALUE { return -1; }
    if append {
        unsafe { ffi::SetFilePointer(h, 0, core::ptr::null_mut(), FILE_END); }
    }
    h as i64
}

pub fn sys_close(fd: i64) -> i64 {
    let r = unsafe { ffi::CloseHandle(fd as isize) };
    if r == 0 { -1 } else { 0 }
}

pub fn sys_write_fd(fd: i64, buf: &[u8]) -> i64 {
    let mut written: u32 = 0;
    let r = unsafe {
        ffi::WriteFile(
            fd as isize,
            buf.as_ptr().cast(),
            buf.len() as u32,
            &mut written as *mut u32,
            core::ptr::null_mut(),
        )
    };
    if r == 0 { -1 } else { written as i64 }
}

pub fn sys_read_fd(fd: i64, buf: &mut [u8]) -> i64 {
    let mut got: u32 = 0;
    let r = unsafe {
        ffi::ReadFile(
            fd as isize,
            buf.as_mut_ptr().cast(),
            buf.len() as u32,
            &mut got as *mut u32,
            core::ptr::null_mut(),
        )
    };
    if r == 0 { -1 } else { got as i64 }
}

pub fn sys_mkdir(path: &[u8], mode: u32) -> i64 {
    let _ = mode;
    let r = unsafe { ffi::CreateDirectoryA(path.as_ptr(), core::ptr::null_mut()) };
    if r == 0 { -1 } else { 0 }
}

pub fn mmap_shared_anon(size: usize) -> *mut u8 {
    let p = unsafe {
        ffi::VirtualAlloc(
            core::ptr::null_mut(),
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        )
    };
    p.cast()
}

pub fn munmap(ptr: *mut u8, size: usize) {
    let _ = size;
    unsafe { ffi::VirtualFree(ptr.cast(), 0, MEM_RELEASE); }
}

pub fn monotonic_ns() -> u64 {
    let mut freq: i64 = 0;
    let mut cnt: i64 = 0;
    unsafe {
        ffi::QueryPerformanceFrequency(&mut freq as *mut i64);
        ffi::QueryPerformanceCounter(&mut cnt as *mut i64);
    }
    if freq <= 0 { return 0; }
    let sec = (cnt / freq) as u64;
    let rem = (cnt % freq) as u64;
    sec.wrapping_mul(1_000_000_000).wrapping_add(rem.wrapping_mul(1_000_000_000) / freq as u64)
}

pub fn fork() -> i64 {
    -1
}

pub fn waitpid(pid: i64) {
    let _ = pid;
}

pub fn exit(code: i32) -> ! {
    unsafe { ffi::ExitProcess(code as u32) }
}

pub fn set_affinity(mask: usize) {
    unsafe {
        let t = ffi::GetCurrentThread();
        ffi::SetThreadAffinityMask(t, mask);
    }
}

#[repr(C, packed)]
pub struct DirEnt64 {
    pub ino: u64,
    pub off: i64,
    pub reclen: u16,
    pub ftype: u8,
}

pub fn sys_getdents64(_fd: i64, _buf: &mut [u8]) -> i64 {
    -1
}
