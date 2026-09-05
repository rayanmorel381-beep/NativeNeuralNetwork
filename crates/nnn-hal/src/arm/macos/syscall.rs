pub const O_RDONLY: i32 = 0;
pub const O_WRONLY: i32 = 1;
pub const O_CREAT: i32 = 0x0200;
pub const O_TRUNC: i32 = 0x0400;
pub const O_APPEND: i32 = 0x0008;

const PROT_READ_WRITE: i32 = 0x3;
const MAP_SHARED: i32 = 0x0001;
const MAP_ANON: i32 = 0x1000;
const CLOCK_UPTIME_RAW: u32 = 8;
const THREAD_AFFINITY_POLICY: u32 = 4;
const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;

mod ffi {
    use core::ffi::c_void;
    #[link(name = "System", kind = "dylib")]
    extern "C" {
        pub fn open(path: *const u8, flags: i32, mode: u32) -> i32;
        pub fn close(fd: i32) -> i32;
        pub fn read(fd: i32, buf: *mut c_void, count: usize) -> isize;
        pub fn write(fd: i32, buf: *const c_void, count: usize) -> isize;
        pub fn mkdir(path: *const u8, mode: u32) -> i32;
        pub fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> i32;
        pub fn clock_gettime_nsec_np(clkid: u32) -> u64;
        pub fn fork() -> i32;
        pub fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
        pub fn _exit(status: i32) -> !;
        pub fn mach_thread_self() -> u32;
        pub fn thread_policy_set(thread: u32, flavor: u32, policy_info: *const i32, count: u32) -> i32;
    }
}

pub fn sys_open(path: &[u8], flags: i32, mode: u32) -> i64 {
    unsafe { ffi::open(path.as_ptr(), flags, mode) as i64 }
}

pub fn sys_close(fd: i64) -> i64 {
    unsafe { ffi::close(fd as i32) as i64 }
}

pub fn sys_write_fd(fd: i64, buf: &[u8]) -> i64 {
    unsafe { ffi::write(fd as i32, buf.as_ptr().cast(), buf.len()) as i64 }
}

pub fn sys_read_fd(fd: i64, buf: &mut [u8]) -> i64 {
    unsafe { ffi::read(fd as i32, buf.as_mut_ptr().cast(), buf.len()) as i64 }
}

pub fn sys_mkdir(path: &[u8], mode: u32) -> i64 {
    unsafe { ffi::mkdir(path.as_ptr(), mode) as i64 }
}

pub fn mmap_shared_anon(size: usize) -> *mut u8 {
    let ptr = unsafe {
        ffi::mmap(
            core::ptr::null_mut(),
            size,
            PROT_READ_WRITE,
            MAP_SHARED | MAP_ANON,
            -1,
            0,
        )
    };
    if ptr as isize == -1 { core::ptr::null_mut() } else { ptr.cast() }
}

pub fn munmap(ptr: *mut u8, size: usize) {
    unsafe { ffi::munmap(ptr.cast(), size); }
}

pub fn monotonic_ns() -> u64 {
    unsafe { ffi::clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
}

pub fn fork() -> i64 {
    unsafe { ffi::fork() as i64 }
}

pub fn waitpid(pid: i64) {
    unsafe { ffi::waitpid(pid as i32, core::ptr::null_mut(), 0); }
}

pub fn exit(code: i32) -> ! {
    unsafe { ffi::_exit(code) }
}

pub fn set_affinity(mask: usize) {
    let tag: i32 = (mask.trailing_zeros() as i32).saturating_add(1);
    unsafe {
        let thread = ffi::mach_thread_self();
        ffi::thread_policy_set(
            thread,
            THREAD_AFFINITY_POLICY,
            &tag as *const i32,
            THREAD_AFFINITY_POLICY_COUNT,
        );
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
