# training/schedulers

Learning-rate schedules: warmup, decays and their combinations.

Façade: `private_api::modules::schedulers`.

## Key types

- `LrSchedule` — the schedule descriptor.
- `ScheduleError` — failure reason.

## Functions

- `compute_learning_rate(...)` — LR at a given step for a schedule.
- `linear_warmup(...)` / `inv_sqrt_warmup(...)` — warmup ramps.
- `cosine_decay(...)` / `step_decay(...)` — decay curves.

## Concrete usage

A classic warmup-then-cosine-decay LR for step `t`:

```rust
use native_neural_network::private_api::modules::schedulers;

fn lr_at(step: usize, warmup: usize, total: usize, base_lr: f32) -> f32 {
    if step < warmup {
        schedulers::linear_warmup(step, warmup, base_lr)
    } else {
        schedulers::cosine_decay(step - warmup, total - warmup, base_lr)
    }
}
```

## Performance levers

- Warmup avoids the early-step instability that large LRs cause on cold weights.
- `inv_sqrt_warmup` is the transformer-standard schedule; `cosine_decay` gives a
  smooth tail toward zero.

## Integration

Feeds the learning rate to [optimizers](../training/optimizers.md) inside the
[trainer](../training/trainer.md) loop (`with_warmup`).
