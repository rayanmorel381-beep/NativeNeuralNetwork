use core::sync::atomic::{AtomicUsize, Ordering};

static ARENA_PTR: AtomicUsize = AtomicUsize::new(0);
static ARENA_CAP: AtomicUsize = AtomicUsize::new(0);
static SLOT_SZ: AtomicUsize = AtomicUsize::new(0);
static SLOT_MAX: AtomicUsize = AtomicUsize::new(0);
static SLOT_USED: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn cache_ensure(n_agents: usize, model_len: usize) -> bool {
    let current = ARENA_PTR.load(Ordering::Acquire);
    if current != 0 {
        let sz = SLOT_SZ.load(Ordering::Relaxed);
        let mx = SLOT_MAX.load(Ordering::Relaxed);
        if sz == model_len && mx == n_agents {
            return true;
        }
        cache_release();
    }
    let capacity = match n_agents.checked_mul(model_len) {
        Some(c) if c > 0 => c,
        _ => return false,
    };
    let ptr = crate::engine::runtime::hardware::mmap_shared_anon(capacity);
    if ptr.is_null() { return false; }
    ARENA_PTR.store(ptr as usize, Ordering::Release);
    ARENA_CAP.store(capacity, Ordering::Release);
    SLOT_SZ.store(model_len, Ordering::Release);
    SLOT_MAX.store(n_agents, Ordering::Release);
    SLOT_USED.store(0, Ordering::Release);
    true
}

pub(crate) fn cache_store(data: &[u8]) -> bool {
    let ptr = ARENA_PTR.load(Ordering::Acquire);
    if ptr == 0 { return false; }
    let used = SLOT_USED.load(Ordering::Acquire);
    let max = SLOT_MAX.load(Ordering::Relaxed);
    let sz = SLOT_SZ.load(Ordering::Relaxed);
    if used >= max || data.len() > sz { return false; }
    unsafe {
        core::ptr::copy_nonoverlapping(
            data.as_ptr(),
            (ptr as *mut u8).add(used * sz),
            data.len(),
        );
    }
    SLOT_USED.store(used + 1, Ordering::Release);
    true
}

pub(crate) fn cache_full() -> bool {
    let used = SLOT_USED.load(Ordering::Acquire);
    let max = SLOT_MAX.load(Ordering::Relaxed);
    max > 0 && used >= max
}

pub(crate) fn compact_f32(
    w_start: usize, w_end: usize,
    b_start: usize, b_end: usize,
    out: &mut [u8],
) {
    let n = SLOT_USED.load(Ordering::Acquire);
    if n < 2 { return; }
    let ptr = ARENA_PTR.load(Ordering::Acquire) as *const u8;
    let sz = SLOT_SZ.load(Ordering::Relaxed);
    let inv = 1.0 / n as f64;
    for off in (w_start..w_end).step_by(4) {
        let mut sum = 0.0f64;
        for i in 0..n {
            let p = unsafe { ptr.add(i * sz + off) };
            let v = f32::from_le_bytes(unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] });
            sum += v as f64;
        }
        let avg = (sum * inv) as f32;
        out[off..off + 4].copy_from_slice(&avg.to_le_bytes());
    }
    for off in (b_start..b_end).step_by(4) {
        let mut sum = 0.0f64;
        for i in 0..n {
            let p = unsafe { ptr.add(i * sz + off) };
            let v = f32::from_le_bytes(unsafe { [*p, *p.add(1), *p.add(2), *p.add(3)] });
            sum += v as f64;
        }
        let avg = (sum * inv) as f32;
        out[off..off + 4].copy_from_slice(&avg.to_le_bytes());
    }
}

pub(crate) fn compact_f64(
    w_start: usize, w_end: usize,
    b_start: usize, b_end: usize,
    out: &mut [u8],
) {
    let n = SLOT_USED.load(Ordering::Acquire);
    if n < 2 { return; }
    let ptr = ARENA_PTR.load(Ordering::Acquire) as *const u8;
    let sz = SLOT_SZ.load(Ordering::Relaxed);
    let inv = 1.0 / n as f64;
    for off in (w_start..w_end).step_by(8) {
        let mut sum = 0.0f64;
        for i in 0..n {
            let p = unsafe { ptr.add(i * sz + off) };
            let v = f64::from_le_bytes(unsafe {
                [*p, *p.add(1), *p.add(2), *p.add(3),
                 *p.add(4), *p.add(5), *p.add(6), *p.add(7)]
            });
            sum += v;
        }
        let avg = sum * inv;
        out[off..off + 8].copy_from_slice(&avg.to_le_bytes());
    }
    for off in (b_start..b_end).step_by(8) {
        let mut sum = 0.0f64;
        for i in 0..n {
            let p = unsafe { ptr.add(i * sz + off) };
            let v = f64::from_le_bytes(unsafe {
                [*p, *p.add(1), *p.add(2), *p.add(3),
                 *p.add(4), *p.add(5), *p.add(6), *p.add(7)]
            });
            sum += v;
        }
        let avg = sum * inv;
        out[off..off + 8].copy_from_slice(&avg.to_le_bytes());
    }
}

pub(crate) fn cache_reset_used() {
    SLOT_USED.store(0, Ordering::Release);
}

pub(crate) fn cache_replica_count(model_len: usize) -> usize {
    let ptr = ARENA_PTR.load(Ordering::Acquire);
    if ptr == 0 { return 0; }
    if SLOT_SZ.load(Ordering::Relaxed) != model_len { return 0; }
    SLOT_USED.load(Ordering::Acquire)
}

pub(crate) fn cache_replica_slice(index: usize, model_len: usize) -> Option<&'static [u8]> {
    let ptr = ARENA_PTR.load(Ordering::Acquire);
    if ptr == 0 { return None; }
    let sz = SLOT_SZ.load(Ordering::Relaxed);
    if sz != model_len { return None; }
    let used = SLOT_USED.load(Ordering::Acquire);
    if index >= used { return None; }
    let base = (ptr as *const u8).wrapping_add(index * sz);
    Some(unsafe { core::slice::from_raw_parts(base, sz) })
}

pub(crate) fn cache_release() {
    let ptr = ARENA_PTR.load(Ordering::Acquire);
    if ptr == 0 { return; }
    let cap = ARENA_CAP.load(Ordering::Relaxed);
    crate::engine::runtime::hardware::munmap(ptr as *mut u8, cap);
    ARENA_PTR.store(0, Ordering::Release);
    ARENA_CAP.store(0, Ordering::Relaxed);
    SLOT_SZ.store(0, Ordering::Relaxed);
    SLOT_MAX.store(0, Ordering::Relaxed);
    SLOT_USED.store(0, Ordering::Relaxed);
}
