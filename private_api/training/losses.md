# training/losses

Loss functions and their gradients — cross-entropy for classification / language
modeling.

Façade: `private_api::modules::losses`.

## Key types

- `LossKind` — selects the loss.
- `LossError` — failure reason.

## Functions

- `cross_entropy(...)` — forward loss from logits/probabilities and targets.
- `cross_entropy_grad(...)` — gradient w.r.t. logits.
- `loss_and_gradient_f32(...)` / `loss_and_gradient_f64(...)` — fused
  forward+backward in one pass.

## Concrete usage

Compute loss and its gradient in a single fused pass (cheaper than two calls):

```rust
use native_neural_network::private_api::modules::losses;

fn step_loss(logits: &[f32], target: usize, grad_out: &mut [f32]) -> f32 {
    losses::loss_and_gradient_f32(logits, target, grad_out)
}
```

## Performance levers

- Prefer the fused `loss_and_gradient_*` over separate `cross_entropy` +
  `cross_entropy_grad` — one traversal instead of two.
- Match precision (`_f32` vs `_f64`) to the model's
  [precision](../numerics/precision.md).

## Integration

Consumed by [trainer](../training/trainer.md); the gradient flows into
[gradients](../training/gradients.md) clipping then
[optimizers](../training/optimizers.md). MoE adds
[moe](../transformer/moe.md) `moe_load_aux_loss`.
