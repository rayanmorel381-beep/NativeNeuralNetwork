use super::{capacity, surge, GuardianPolicy};

fn within_limit(current: u64, requested: u64, limit_pct: u8) -> bool {
    current.saturating_add(requested) <= u64::from(limit_pct)
}

pub(crate) fn cpu(policy: GuardianPolicy, current_pct: u64, requested_pct: u64) -> bool {
    within_limit(current_pct, requested_pct, policy.cpu_limit_pct()) && surge::allow_cpu(requested_pct)
}

pub(crate) fn memory(policy: GuardianPolicy, current_pct: u64, requested_pct: u64) -> bool {
    let current = capacity::memory_usage().unwrap_or(current_pct);
    within_limit(current, requested_pct, policy.ram_limit_pct()) && surge::allow_memory(requested_pct)
}

pub(crate) fn accelerator(policy: GuardianPolicy, backend: super::Accelerator, current_pct: u64, requested_pct: u64) -> bool {
    let limit = match backend {
        super::Accelerator::Gpu => policy.gpu_limit_pct(),
        super::Accelerator::Tpu => policy.tpu_limit_pct(),
        super::Accelerator::Lpu => policy.lpu_limit_pct(),
    };
    within_limit(current_pct, requested_pct, limit)
}
