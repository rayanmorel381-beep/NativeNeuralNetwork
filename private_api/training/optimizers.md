# training/optimizers

Parameter update rules: AdamW and SGD, with their state and step functions.

Façade: `private_api::modules::optimizers`.

## Key types

- `OptimizerKind` — selects SGD vs AdamW.
- `AdamwConfig` — AdamW hyperparameters (betas, weight decay, eps).
- `OptimizerError` — failure reason.

## Functions

- `adamw(...)` / `step_adamw(...)` — AdamW update.
- `step_sgd(...)` — SGD update.
- `apply_optimizer_step(...)` — dispatch on `OptimizerKind`.
- `optimizer_state_len(...)` — size the optimizer state buffer.

## Concrete usage

Size the AdamW moment buffers once, then apply an update per step:

```rust
use native_neural_network::private_api::modules::optimizers::{
    self, AdamwConfig, OptimizerKind,
};

let n_params = 73_728;
let state_len = optimizers::optimizer_state_len(OptimizerKind::AdamW, n_params);
let mut state = vec![0.0f32; state_len];

let cfg = AdamwConfig::default();
fn update(cfg: &AdamwConfig, w: &mut [f32], g: &[f32], state: &mut [f32], lr: f32) {
    optimizers::step_adamw(cfg, w, g, state, lr);
}
```

## Performance levers

- AdamW converges more robustly; SGD needs less state (`optimizer_state_len` is
  smaller) and is faster per step.
- Allocate `state` once with `optimizer_state_len`; reuse across all steps.

## Integration

Called by [trainer](../training/trainer.md) after
[gradients](../training/gradients.md) are accumulated; learning rate comes from
[schedulers](../training/schedulers.md).
