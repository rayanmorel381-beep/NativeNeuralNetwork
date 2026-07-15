# geometry/conv5d

5D convolution with both forward and backward passes — the higher-dimensional
counterpart used by the crate's spatial representation.

Façade: `private_api::modules::conv5d`.

## Key types

- `Conv5dOutputShapeArgs` — inputs to output-shape math.
- `Conv5dBackwardArgs` — inputs to the gradient pass.

## Functions

- `conv5d_forward(...)` — forward convolution.
- `conv5d_backward(...)` — gradient w.r.t. inputs/kernel.
- `conv5d_output_shape(...)` / `conv5d_output_dim(...)` — output extents.
- `conv5d_is_compatible(...)` / `conv5d_layout_compatible(...)` — layout checks.

## Concrete usage

Forward then backward, reusing the validated shape:

```rust
use native_neural_network::private_api::modules::conv5d::{
    self, Conv5dOutputShapeArgs, Conv5dBackwardArgs,
};

fn forward(args: &Conv5dOutputShapeArgs, x: &[f32], w: &[f32], y: &mut [f32]) {
    assert!(conv5d::conv5d_is_compatible(args));
    conv5d::conv5d_forward(args, x, w, y);
}

fn backward(bargs: &Conv5dBackwardArgs, dy: &[f32], dx: &mut [f32], dw: &mut [f32]) {
    conv5d::conv5d_backward(bargs, dy, dx, dw);
}
```

## Performance levers

- Validate with `conv5d_is_compatible` at setup; skip re-checking in the loop.
- Backward is the expensive half — only compute it during
  [trainer](../training/trainer.md) steps, never at inference.

## Integration

Configured by [conv_net](../training/conv_net.md); geometry feeds
[sphere5d](../geometry/sphere5d.md).
