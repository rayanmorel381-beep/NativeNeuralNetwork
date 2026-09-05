use core::sync::atomic::{AtomicU64, Ordering};

static CPU_BUDGET: AtomicU64 = AtomicU64::new(0);
static MEMORY_BUDGET: AtomicU64 = AtomicU64::new(0);
static CPU_USED: AtomicU64 = AtomicU64::new(0);
static MEMORY_USED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_cpu_budget(budget: u64) { CPU_BUDGET.store(budget, Ordering::Release); CPU_USED.store(0, Ordering::Release); }
pub(crate) fn set_memory_budget(budget: u64) { MEMORY_BUDGET.store(budget, Ordering::Release); MEMORY_USED.store(0, Ordering::Release); }

fn consume(budget: &AtomicU64, used: &AtomicU64, amount: u64) -> bool {
    let limit = budget.load(Ordering::Acquire);
    if limit == 0 { return true; }
    loop {
        let current = used.load(Ordering::Acquire);
        let next = current.saturating_add(amount);
        if next > limit { return false; }
        if used.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed).is_ok() { return true; }
    }
}

pub(crate) fn allow_cpu(amount: u64) -> bool { consume(&CPU_BUDGET, &CPU_USED, amount) }
pub(crate) fn allow_memory(amount: u64) -> bool { consume(&MEMORY_BUDGET, &MEMORY_USED, amount) }
