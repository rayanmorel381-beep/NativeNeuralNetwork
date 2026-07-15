# transformer/normalization

Layer normalization primitives. The crate uses RMSNorm, the norm of choice for
modern transformers.

Façade: `private_api::modules::normalization`.

## Key types

- `NormError` — failure reason.

## Functions

- `rms_norm_in_place(...)` — root-mean-square normalize a row in place.

## Concrete usage

Normalize a hidden-state row before a transformer sub-block, scaled by learned
weights, without any extra buffer:

```rust
use native_neural_network::private_api::modules::normalization;

fn pre_norm(hidden: &mut [f32], gamma: &[f32], eps: f32) {
    normalization::rms_norm_in_place(hidden, gamma, eps);
}
```

## Performance levers

- In-place operation means no extra buffer on the
  [inference](../transformer/inference.md) hot path.
- RMSNorm skips the mean-subtraction of classic LayerNorm, saving a pass.

## Integration

Applied before/after transformer sub-blocks in
[inference](../transformer/inference.md).
