use core::sync::atomic::{AtomicUsize, Ordering};

pub type ParallelForFn = fn(usize, &(dyn Fn(usize) + Sync));

static EXECUTOR: AtomicUsize = AtomicUsize::new(0);
static THRESHOLD: AtomicUsize = AtomicUsize::new(1 << 16);

pub fn set_parallel_executor(executor: ParallelForFn) {
    EXECUTOR.store(executor as usize, Ordering::Release);
}

pub fn clear_parallel_executor() {
    EXECUTOR.store(0, Ordering::Release);
}

pub fn set_parallel_threshold(work: usize) {
    THRESHOLD.store(work.max(1), Ordering::Release);
}

pub fn configure_parallel_executor(executor: ParallelForFn) {
    set_parallel_executor(executor)
}

pub fn reset_parallel_executor() {
    clear_parallel_executor()
}

pub fn configure_parallel_threshold(work: usize) {
    set_parallel_threshold(work)
}

pub fn get_parallel_threshold() -> usize {
    THRESHOLD.load(Ordering::Acquire)
}

fn default_parallel_executor(n: usize, body: &(dyn Fn(usize) + Sync)) {
    for i in 0..n {
        body(i);
    }
}

pub fn initialize_parallel_runtime() {
    configure_parallel_executor(default_parallel_executor);
    configure_parallel_threshold(1 << 16);
}

fn executor() -> Option<ParallelForFn> {
    let raw = EXECUTOR.load(Ordering::Acquire);
    if raw == 0 {
        None
    } else {
        Some(unsafe { core::mem::transmute::<usize, ParallelForFn>(raw) })
    }
}

pub(crate) fn parallel_for(n: usize, cost_per_index: usize, body: &(dyn Fn(usize) + Sync)) {
    if n == 0 {
        return;
    }
    let total = n.saturating_mul(cost_per_index.max(1));
    let threshold = get_parallel_threshold();
    if n > 1 && total >= threshold {
        if let Some(run) = executor() {
            run(n, body);
            return;
        }
        reset_parallel_executor();
    }
    for i in 0..n {
        body(i);
    }
}

pub(crate) struct SyncMutPtr<T>(pub *mut T);

unsafe impl<T> Sync for SyncMutPtr<T> {}
unsafe impl<T> Send for SyncMutPtr<T> {}

impl<T> Clone for SyncMutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SyncMutPtr<T> {}
