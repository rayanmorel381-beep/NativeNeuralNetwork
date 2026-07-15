# runtime/runtime

Hardware profiling, the CPU/RAM consumption guard, and the raw syscall layer.
This is the module that keeps the crate from saturating the machine — Veyma's
stability on 32 GB comes from here.

Façade: `private_api::modules::runtime` (maps to `crate::runtime::hardware`).

## Key types

- `ConsumptionGuard` — the CPU/RAM budget gate.
- `HardwareProfile` / `BackendCapabilities` / `ResourceSnapshot` — detected
  hardware and live usage.
- `RuntimeProfile` / `RuntimeEstimate` / `RuntimeFlopsEstimate` — cost estimates.
- `BudgetFit` — result of a budget check.
- `FixedSliceVec` — a `no_std` bounded vector.
- `ParallelForFn` — the parallel-map function type.
- `RuntimeError` — failure reason.

## Key constants

- `COMPUTE_CAP_PPM` — CPU cap in parts-per-million (**800_000 = 80 %**).
- `RAM_CAP_PPM` — RAM cap (**700_000 = 70 %**).
- Syscall numbers (`SYS_*`), file flags (`O_*`), memory attrs (`ATTR_*`) for the
  Linux path; Windows equivalents (`CreateFileA`, `ReadFile`, …) for portability.

## Concrete usage

Before parallelizing heavy work, ask the guard how much you're actually allowed
to use — this is the exact pattern the fork pool uses:

```rust
use native_neural_network::private_api::modules::runtime::ConsumptionGuard;

let guard = ConsumptionGuard::detect();

// RAM: how many worker slots fit in the 70 % budget?
let budget = guard.ram_budget_bytes();
let slot_size = 32 * 1024 * 1024;
let mut workers = requested_workers;
if budget > 0 && slot_size > 0 {
    workers = workers.min((budget / slot_size).max(1));
}

// CPU: clamp worker count to stay under the 80 % compute cap.
let workers = guard.clamp_workers(workers);
```

## Performance levers

- Always route parallelism through `clamp_workers` / `ram_fits` — never spawn a
  fixed number of threads blind.
- Idle fork workers block on a futex → **0 %** CPU while waiting (no busy-wait).
- Use `ResourceSnapshot` to observe live pressure and back off adaptively.

## Integration

Consulted by [trainer](../training/trainer.md) (worker pool),
[engine](../core/engine.md) (backend headroom) and
[model_config](../core/model_config.md) `ResourcePolicy`.
