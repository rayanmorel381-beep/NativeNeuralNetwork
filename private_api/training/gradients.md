# training/gradients

Gradient hygiene: global-norm clipping, norm computation and NaN/Inf guards that
keep training from diverging.

Façade: `private_api::modules::gradients`.

## Key types

- `GradientError` — failure reason.

## Functions

- `clip_by_global_norm(...)` — rescale all grads so their global L2 norm ≤ max.
- `l2_norm(...)` — global L2 norm.
- `all_finite(...)` — true if every value is finite.
- `has_nan_f32` / `has_nan_f64` / `has_inf_f32` / `has_inf_f64` — targeted checks.
- `within_abs_bound_f32` / `within_abs_bound_f64` — magnitude bound check.

## Concrete usage

Guard a step: bail on non-finite grads, then clip before applying:

```rust
use native_neural_network::private_api::modules::gradients;

fn safe_step(grads: &mut [f32], max_norm: f32) -> bool {
    if !gradients::all_finite(grads) {
        return false;                       // skip this step, don't corrupt weights
    }
    gradients::clip_by_global_norm(grads, max_norm);
    true
}
```

## Performance levers

- `all_finite` is cheaper than separate `has_nan`/`has_inf`; use it as the fast
  gate, then diagnose with the targeted checks only on failure.
- Global-norm clipping stabilizes deep stacks without per-parameter tuning.

## Integration

Applied by [trainer](../training/trainer.md) between
[losses](../training/losses.md) backprop and
[optimizers](../training/optimizers.md) steps.
