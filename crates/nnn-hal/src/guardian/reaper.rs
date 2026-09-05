use core::sync::atomic::{AtomicU64, Ordering};

static LAST_ACTIVITY_NS: AtomicU64 = AtomicU64::new(0);
static IDLE_TIMEOUT_NS: AtomicU64 = AtomicU64::new(30_000_000_000);

pub(crate) fn mark_activity(timestamp_ns: u64) { LAST_ACTIVITY_NS.store(timestamp_ns, Ordering::Release); }
pub(crate) fn expired(now_ns: u64) -> bool {
    let last = LAST_ACTIVITY_NS.load(Ordering::Acquire);
    let timeout = IDLE_TIMEOUT_NS.load(Ordering::Acquire);
    last != 0 && timeout != 0 && now_ns.saturating_sub(last) >= timeout
}
