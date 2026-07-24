use core::sync::atomic::{AtomicU32, AtomicUsize, Ordering, fence};
use crate::engine::runtime::hardware;

const SLOT_STRIDE: usize = 64;
const STATE_IDLE: u32 = 0;
const STATE_WORK: u32 = 1;
const STATE_DONE: u32 = 2;
const STATE_EXIT: u32 = 0xFF;
const SPIN_LIMIT: u32 = 4096;
static POOL_CTRL: AtomicUsize = AtomicUsize::new(0);
static POOL_N: AtomicUsize = AtomicUsize::new(0);
static POOL_STAGE: AtomicUsize = AtomicUsize::new(0);
static POOL_STAGE_CAP: AtomicUsize = AtomicUsize::new(0);

#[cfg(any(target_os = "linux", target_os = "android"))]
#[inline]
fn worker_park(slot_u32: *const u32, val: u32) {
    hardware::futex_wait_u32(slot_u32, val);
}
#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[inline]
fn worker_park(_slot_u32: *const u32, _val: u32) {
    core::hint::spin_loop();
}

#[cfg(any(target_os = "linux", target_os = "android"))]
#[inline]
fn worker_unpark(slot_u32: *const u32) {
    hardware::futex_wake_u32(slot_u32, 1);
}
#[cfg(not(any(target_os = "linux", target_os = "android")))]
#[inline]
fn worker_unpark(_slot_u32: *const u32) {}

pub(crate) struct ParPool {
    ctrl: *mut u8,
    ctrl_size: usize,
    stage: *mut u8,
    stage_cap: usize,
    pids: [i64; 32],
    pub(crate) n_workers: usize,
}

impl ParPool {
    pub(crate) fn empty() -> Self {
        Self { ctrl: core::ptr::null_mut(), ctrl_size: 0, stage: core::ptr::null_mut(), stage_cap: 0, pids: [0; 32], n_workers: 0 }
    }
    pub(crate) fn is_active(&self) -> bool { self.n_workers > 0 }
}

fn worker_loop(idx: usize, n_workers: usize, ctrl: *mut u8) -> ! {
    let slot = unsafe { ctrl.add(idx * SLOT_STRIDE) };
    let state = unsafe { &*(slot as *const AtomicU32) };
    let slot_u32 = slot as *const u32;
    loop {
        let st = state.load(Ordering::Acquire);
        if st == STATE_EXIT {
            hardware::exit(0);
        }
        if st != STATE_WORK {
            let mut spins = 0u32;
            let mut current = st;
            while current != STATE_WORK && current != STATE_EXIT && spins < SPIN_LIMIT {
                core::hint::spin_loop();
                current = state.load(Ordering::Acquire);
                spins += 1;
            }
            if current == STATE_EXIT {
                hardware::exit(0);
            }
            if current != STATE_WORK {
                worker_park(slot_u32, current);
                continue;
            }
        }
        let start   = unsafe { core::ptr::read_volatile(slot.add(8)  as *const usize) };
        let end     = unsafe { core::ptr::read_volatile(slot.add(16) as *const usize) };
        let fn_data = unsafe { core::ptr::read_volatile(slot.add(24) as *const usize) };
        let fn_vtab = unsafe { core::ptr::read_volatile(slot.add(32) as *const usize) };
        fence(Ordering::Acquire);
        let body: &(dyn Fn(usize) + Sync) = unsafe {
            core::mem::transmute::<(usize, usize), &(dyn Fn(usize) + Sync)>((fn_data, fn_vtab))
        };
        for i in start..end {
            body(i);
        }
        state.store(STATE_DONE, Ordering::Release);
        let barrier_ptr = unsafe { ctrl.add(n_workers * SLOT_STRIDE) };
        let barrier = unsafe { &*(barrier_ptr as *const AtomicU32) };
        barrier.fetch_add(1, Ordering::AcqRel);
        worker_unpark(barrier_ptr as *const u32);
    }
}

pub(crate) fn par_pool_create(n_workers: usize) -> ParPool {
    let n = n_workers.min(32);
    if n < 2 { return ParPool::empty(); }
    let ctrl_size = (n + 1) * SLOT_STRIDE;
    let ctrl = hardware::mmap_shared_anon(ctrl_size);
    if ctrl.is_null() { return ParPool::empty(); }
    unsafe { core::ptr::write_bytes(ctrl, 0, ctrl_size); }
    let l2 = crate::engine::runtime::l2_cache_bytes();
    let stage_cap = if l2 >= 512 * 1024 { 16384 }
        else if l2 >= 256 * 1024 { 8192 }
        else if l2 > 0 { 4096 }
        else { 4096 };
    let stage = hardware::mmap_shared_anon(stage_cap);
    if stage.is_null() {
        hardware::munmap(ctrl, ctrl_size);
        return ParPool::empty();
    }
    unsafe { core::ptr::write_bytes(stage, 0, stage_cap); }
    let mut pool = ParPool { ctrl, ctrl_size, stage, stage_cap, pids: [0; 32], n_workers: 0 };
    for i in 0..n {
        let pid = hardware::fork();
        if pid < 0 {
            for j in 0..pool.n_workers {
                unsafe { (*(pool.ctrl.add(j * SLOT_STRIDE) as *const AtomicU32)).store(STATE_EXIT, Ordering::Release); }
            }
            for j in 0..pool.n_workers { hardware::waitpid(pool.pids[j]); }
            hardware::munmap(stage, stage_cap);
            hardware::munmap(ctrl, ctrl_size);
            return ParPool::empty();
        }
        if pid == 0 {            #[cfg(any(target_os = "linux", target_os = "android"))]
            hardware::prctl_set_pdeathsig(9);            worker_loop(i, n, ctrl);
        }
        pool.pids[i] = pid;
        pool.n_workers += 1;
    }
    pool
}

pub(crate) fn par_pool_launch() -> ParPool {
    if crate::engine::runtime::install_as_parallel_executor() {
        return ParPool::empty();
    }
    let guard = crate::engine::runtime::ConsumptionGuard::detect();
    let cores = guard.cpu_workers().max(1);
    let pool = par_pool_create(cores);
    if pool.is_active() {
        par_pool_install(&pool);
    }
    pool
}

pub(crate) fn par_pool_install(pool: &ParPool) {
    POOL_CTRL.store(pool.ctrl as usize, Ordering::Release);
    POOL_N.store(pool.n_workers, Ordering::Release);
    POOL_STAGE.store(pool.stage as usize, Ordering::Release);
    POOL_STAGE_CAP.store(pool.stage_cap, Ordering::Release);
    crate::engine::runtime::set_parallel_executor(par_pool_run);
}

pub(crate) fn par_pool_destroy(pool: &ParPool) {
    crate::engine::runtime::reset_parallel_executor();
    POOL_CTRL.store(0, Ordering::Release);
    POOL_N.store(0, Ordering::Release);
    POOL_STAGE.store(0, Ordering::Release);
    POOL_STAGE_CAP.store(0, Ordering::Release);
    if !pool.is_active() {
        return;
    }
    for i in 0..pool.n_workers {
        unsafe { (*(pool.ctrl.add(i * SLOT_STRIDE) as *const AtomicU32)).store(STATE_EXIT, Ordering::Release); }
        worker_unpark(unsafe { pool.ctrl.add(i * SLOT_STRIDE) } as *const u32);
    }
    for i in 0..pool.n_workers {
        hardware::waitpid(pool.pids[i]);
    }
    hardware::munmap(pool.stage, pool.stage_cap);
    hardware::munmap(pool.ctrl, pool.ctrl_size);
}

fn par_pool_run(n: usize, body: &(dyn Fn(usize) + Sync)) {
    let ctrl = POOL_CTRL.load(Ordering::Acquire) as *mut u8;
    let n_workers = POOL_N.load(Ordering::Acquire);
    let stage = POOL_STAGE.load(Ordering::Acquire) as *mut u8;
    if ctrl.is_null() || n_workers == 0 || stage.is_null() {
        for i in 0..n { body(i); }
        return;
    }
    let (fn_data, fn_vtab): (usize, usize) = unsafe {
        core::mem::transmute::<&(dyn Fn(usize) + Sync), (usize, usize)>(body)
    };
    let env_size = unsafe { *((fn_vtab as *const usize).add(1)) };
    let stage_cap = POOL_STAGE_CAP.load(Ordering::Acquire);
    if stage_cap == 0 || env_size > stage_cap {
        for i in 0..n { body(i); }
        return;
    }
    unsafe { core::ptr::copy_nonoverlapping(fn_data as *const u8, stage, env_size); }
    fence(Ordering::Release);
    let shared_data = stage as usize;
    let barrier_ptr = unsafe { ctrl.add(n_workers * SLOT_STRIDE) };
    let barrier = unsafe { &*(barrier_ptr as *const AtomicU32) };
    barrier.store(0, Ordering::Release);
    fence(Ordering::Release);
    let base = n / n_workers;
    let rem = n % n_workers;
    let mut active = 0usize;
    for w in 0..n_workers {
        let start = w * base + if w < rem { w } else { rem };
        let end = start + base + if w < rem { 1 } else { 0 };
        if start >= end { continue; }
        let slot = unsafe { ctrl.add(w * SLOT_STRIDE) };
        unsafe {
            core::ptr::write_volatile(slot.add(8)  as *mut usize, start);
            core::ptr::write_volatile(slot.add(16) as *mut usize, end);
            core::ptr::write_volatile(slot.add(24) as *mut usize, shared_data);
            core::ptr::write_volatile(slot.add(32) as *mut usize, fn_vtab);
        }
        fence(Ordering::Release);
        unsafe { (*(slot as *const AtomicU32)).store(STATE_WORK, Ordering::Release); }
        worker_unpark(slot as *const u32);
        active += 1;
    }
    let barrier_u32 = barrier_ptr as *const u32;
    loop {
        let done = barrier.load(Ordering::Acquire);
        if done >= active as u32 { break; }
        let mut spins = 0u32;
        let mut current = done;
        while (current as usize) < active && spins < SPIN_LIMIT {
            core::hint::spin_loop();
            current = barrier.load(Ordering::Acquire);
            spins += 1;
        }
        if (current as usize) >= active { break; }
        worker_park(barrier_u32, current);
    }
    for w in 0..active {
        let slot = unsafe { ctrl.add(w * SLOT_STRIDE) };
        unsafe { (*(slot as *const AtomicU32)).store(STATE_IDLE, Ordering::Release); }
        worker_unpark(slot as *const u32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::runtime::{parallel_for, SyncMutPtr};

    fn expected(i: usize) -> u64 {
        (i as u64).wrapping_mul(2_654_435_761).wrapping_add(1)
    }

    #[test]
    fn pool_executes_parallel_closure_in_shared_memory() {
        let n = 4096usize;
        let bytes = n * core::mem::size_of::<u64>();
        let base = hardware::mmap_shared_anon(bytes);
        assert!(!base.is_null(), "mmap partagé a échoué");
        let out = base as *mut u64;
        unsafe { core::ptr::write_bytes(out, 0, n); }

        let pool = par_pool_create(4);
        if !pool.is_active() {
            hardware::munmap(base, bytes);
            return;
        }
        par_pool_install(&pool);
        let out_ptr = SyncMutPtr(out);
        parallel_for(n, 1024, &move |i| {
            let _ = &out_ptr;
            unsafe { *out_ptr.0.add(i) = expected(i); }
        });
        par_pool_destroy(&pool);

        let mut wrong = 0usize;
        for i in 0..n {
            if unsafe { *out.add(i) } != expected(i) {
                wrong += 1;
            }
        }
        hardware::munmap(base, bytes);
        assert_eq!(wrong, 0, "les workers forkés n'ont pas produit le résultat attendu");
    }

    #[test]
    fn pool_matches_sequential_reference() {
        let n = 2048usize;
        let bytes = n * core::mem::size_of::<u64>();
        let base = hardware::mmap_shared_anon(bytes);
        assert!(!base.is_null());
        let out = base as *mut u64;
        unsafe { core::ptr::write_bytes(out, 0, n); }

        let mut reference = [0u64; 2048];
        for (i, r) in reference.iter_mut().enumerate() {
            *r = expected(i);
        }

        let pool = par_pool_create(4);
        if !pool.is_active() {
            hardware::munmap(base, bytes);
            return;
        }
        par_pool_install(&pool);
        let out_ptr = SyncMutPtr(out);
        parallel_for(n, 4096, &move |i| {
            let _ = &out_ptr;
            unsafe { *out_ptr.0.add(i) = expected(i); }
        });
        par_pool_destroy(&pool);

        let mut ok = true;
        for i in 0..n {
            if unsafe { *out.add(i) } != reference[i] {
                ok = false;
                break;
            }
        }
        hardware::munmap(base, bytes);
        assert!(ok, "sortie parallèle différente de la référence séquentielle");
    }
}
