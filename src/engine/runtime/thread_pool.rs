use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering, fence};
use crate::engine::runtime::hardware;

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const STACK_SIZE: usize = 2 * 1024 * 1024;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const MAX_THREADS: usize = 64;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const TP_SLOT_STRIDE: usize = 64;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const TP_BARRIER_OFFSET: usize = MAX_THREADS * TP_SLOT_STRIDE;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const TP_STATE_IDLE: u32 = 0;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const TP_STATE_WORK: u32 = 1;

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
const CLONE_THREAD_FLAGS: usize = 0x0005_0F00;

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
static TP_CTRL: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
static TP_PARTICIPANTS: AtomicUsize = AtomicUsize::new(0);
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
static TP_INIT: AtomicU32 = AtomicU32::new(0);
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
static TP_WAKE_SINK: AtomicU32 = AtomicU32::new(0);

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
fn thread_workers() -> usize {
    crate::engine::runtime::ConsumptionGuard::detect().cpu_workers().max(1)
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
#[repr(C)]
struct TpWorkerArg {
    idx: usize,
    ctrl: usize,
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
extern "C" fn tp_worker_entry(arg: usize) {
    let wa = unsafe { &*(arg as *const TpWorkerArg) };
    let ctrl = wa.ctrl as *mut u8;
    let slot = unsafe { ctrl.add(wa.idx * TP_SLOT_STRIDE) };
    let state = unsafe { &*(slot as *const AtomicU32) };
    let slot_u32 = slot as *const u32;
    let barrier_ptr = unsafe { ctrl.add(TP_BARRIER_OFFSET) };
    let barrier = unsafe { &*(barrier_ptr as *const AtomicU32) };
    loop {
        let st = state.load(Ordering::Acquire);
        if st != TP_STATE_WORK {
            hardware::futex_wait_u32(slot_u32, st);
            continue;
        }
        let start = unsafe { core::ptr::read_volatile(slot.add(8) as *const usize) };
        let end = unsafe { core::ptr::read_volatile(slot.add(16) as *const usize) };
        let fn_data = unsafe { core::ptr::read_volatile(slot.add(24) as *const usize) };
        let fn_vtab = unsafe { core::ptr::read_volatile(slot.add(32) as *const usize) };
        fence(Ordering::Acquire);
        let body: &(dyn Fn(usize) + Sync) = unsafe {
            core::mem::transmute::<(usize, usize), &(dyn Fn(usize) + Sync)>((fn_data, fn_vtab))
        };
        for i in start..end {
            body(i);
        }
        state.store(TP_STATE_IDLE, Ordering::Release);
        barrier.fetch_add(1, Ordering::AcqRel);
        hardware::futex_wake_u32(barrier_ptr as *const u32, 1);
    }
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
fn tp_pool_init() -> bool {
    match TP_INIT.load(Ordering::Acquire) {
        2 => return true,
        3 => return false,
        _ => {}
    }
    if TP_INIT
        .compare_exchange(0, 1, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        loop {
            match TP_INIT.load(Ordering::Acquire) {
                2 => return true,
                3 => return false,
                _ => core::hint::spin_loop(),
            }
        }
    }
    let target = thread_workers().min(MAX_THREADS).max(1);
    if target < 2 {
        TP_INIT.store(3, Ordering::Release);
        return false;
    }
    let ctrl_size = (MAX_THREADS + 1) * TP_SLOT_STRIDE;
    let ctrl = hardware::mmap_shared_anon(ctrl_size);
    if ctrl.is_null() {
        TP_INIT.store(3, Ordering::Release);
        return false;
    }
    unsafe { core::ptr::write_bytes(ctrl, 0, ctrl_size); }
    let args_size = MAX_THREADS * core::mem::size_of::<TpWorkerArg>();
    let args = hardware::mmap_shared_anon(args_size);
    if args.is_null() {
        hardware::munmap(ctrl, ctrl_size);
        TP_INIT.store(3, Ordering::Release);
        return false;
    }
    let mut spawned = 0usize;
    for idx in 1..target {
        let stack = hardware::mmap_shared_anon(STACK_SIZE);
        if stack.is_null() {
            break;
        }
        let ap = unsafe { (args as *mut TpWorkerArg).add(idx) };
        unsafe { core::ptr::write(ap, TpWorkerArg { idx, ctrl: ctrl as usize }); }
        fence(Ordering::Release);
        let stack_top = stack as usize + STACK_SIZE;
        let tid = unsafe {
            hardware::spawn_thread(
                CLONE_THREAD_FLAGS,
                stack_top,
                tp_worker_entry,
                ap as usize,
                &TP_WAKE_SINK as *const AtomicU32 as usize,
            )
        };
        if tid <= 0 {
            hardware::munmap(stack, STACK_SIZE);
            break;
        }
        spawned += 1;
    }
    if spawned == 0 {
        hardware::munmap(args, args_size);
        hardware::munmap(ctrl, ctrl_size);
        TP_INIT.store(3, Ordering::Release);
        return false;
    }
    TP_CTRL.store(ctrl as usize, Ordering::Release);
    TP_PARTICIPANTS.store(spawned + 1, Ordering::Release);
    TP_INIT.store(2, Ordering::Release);
    true
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
pub(crate) fn thread_parallel_for(n: usize, body: &(dyn Fn(usize) + Sync)) {
    if n == 0 {
        return;
    }
    if !tp_pool_init() {
        for i in 0..n {
            body(i);
        }
        return;
    }
    let participants = TP_PARTICIPANTS.load(Ordering::Acquire);
    let workers = participants.min(n);
    if workers < 2 {
        for i in 0..n {
            body(i);
        }
        return;
    }
    let ctrl = TP_CTRL.load(Ordering::Acquire) as *mut u8;
    let (fn_data, fn_vtab): (usize, usize) = unsafe {
        core::mem::transmute::<&(dyn Fn(usize) + Sync), (usize, usize)>(body)
    };
    let barrier_ptr = unsafe { ctrl.add(TP_BARRIER_OFFSET) };
    let barrier = unsafe { &*(barrier_ptr as *const AtomicU32) };
    barrier.store(0, Ordering::Release);
    fence(Ordering::Release);
    let base = n / workers;
    let rem = n % workers;
    let mut active = 0usize;
    for w in 1..workers {
        let start = w * base + if w < rem { w } else { rem };
        let end = start + base + if w < rem { 1 } else { 0 };
        if start >= end {
            continue;
        }
        let slot = unsafe { ctrl.add(w * TP_SLOT_STRIDE) };
        unsafe {
            core::ptr::write_volatile(slot.add(8) as *mut usize, start);
            core::ptr::write_volatile(slot.add(16) as *mut usize, end);
            core::ptr::write_volatile(slot.add(24) as *mut usize, fn_data);
            core::ptr::write_volatile(slot.add(32) as *mut usize, fn_vtab);
        }
        fence(Ordering::Release);
        unsafe { (*(slot as *const AtomicU32)).store(TP_STATE_WORK, Ordering::Release); }
        hardware::futex_wake_u32(slot as *const u32, 1);
        active += 1;
    }
    let p_end = base + if rem > 0 { 1 } else { 0 };
    for i in 0..p_end {
        body(i);
    }
    let barrier_u32 = barrier_ptr as *const u32;
    loop {
        let done = barrier.load(Ordering::Acquire);
        if done >= active as u32 {
            break;
        }
        hardware::futex_wait_u32(barrier_u32, done);
    }
    fence(Ordering::Acquire);
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android")))]
pub(crate) fn install_as_parallel_executor() -> bool {
    if tp_pool_init() {
        crate::engine::runtime::set_parallel_executor(thread_parallel_for);
        true
    } else {
        false
    }
}

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android"))))]
pub(crate) fn install_as_parallel_executor() -> bool {
    false
}

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "android"))))]
pub(crate) fn thread_parallel_for(n: usize, body: &(dyn Fn(usize) + Sync)) {
    for i in 0..n {
        body(i);
    }
}

pub(crate) fn parallel_memcpy(dst: &mut [u8], src: &[u8]) {
    const PAR_THRESHOLD: usize = 8 * 1024 * 1024;
    if dst.len() != src.len() || dst.len() < PAR_THRESHOLD {
        dst.copy_from_slice(src);
        return;
    }
    let len = dst.len();
    let dst_addr = dst.as_mut_ptr() as usize;
    let src_addr = src.as_ptr() as usize;
    const CHUNK: usize = 4 * 1024 * 1024;
    let n_chunks = len.div_ceil(CHUNK);
    thread_parallel_for(n_chunks, &|c| {
        let start = c * CHUNK;
        let end = (start + CHUNK).min(len);
        unsafe {
            core::ptr::copy_nonoverlapping(
                (src_addr as *const u8).add(start),
                (dst_addr as *mut u8).add(start),
                end - start,
            );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::runtime::SyncMutPtr;

    fn expected(i: usize) -> u64 {
        (i as u64).wrapping_mul(2_654_435_761).wrapping_add(7)
    }

    #[test]
    fn threads_write_private_address_space() {
        const N: usize = 4096;
        let mut data = [0u64; N];
        let out = SyncMutPtr(data.as_mut_ptr());
        thread_parallel_for(N, &move |i| {
            let _ = &out;
            unsafe { *out.0.add(i) = expected(i); }
        });
        let mut wrong = 0usize;
        for i in 0..N {
            if data[i] != expected(i) {
                wrong += 1;
            }
        }
        assert_eq!(wrong, 0, "les threads n'ont pas ecrit dans l'espace memoire partage");
    }

    #[test]
    fn threads_match_sequential_reference() {
        const N: usize = 3000;
        let mut data = [0u64; N];
        let mut reference = [0u64; N];
        for (i, r) in reference.iter_mut().enumerate() {
            *r = expected(i);
        }
        let out = SyncMutPtr(data.as_mut_ptr());
        thread_parallel_for(N, &move |i| {
            let _ = &out;
            unsafe { *out.0.add(i) = expected(i); }
        });
        let mut ok = true;
        for i in 0..N {
            if data[i] != reference[i] {
                ok = false;
            }
        }
        assert!(ok, "resultat threads different de la reference sequentielle");
    }
}
