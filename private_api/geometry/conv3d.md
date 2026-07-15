# geometry/conv3d

3D convolution: forward pass, output-shape computation and layout compatibility
checks.

Façade: `private_api::modules::conv3d`.

## Key types

- `Conv3dOutputShapeArgs` — the inputs to output-shape math (dims, kernel,
  stride, padding).

## Functions

- `conv3d_forward(...)` — the forward convolution.
- `conv3d_output_shape(...)` / `conv3d_output_dim(...)` — compute output extents.
- `conv3d_is_compatible(...)` / `conv3d_layout_compatible(...)` — validate a
  layout before running.

## Concrete usage

Compute the output shape first, size the output buffer, then run the forward
pass:

```rust
use native_neural_network::private_api::modules::conv3d::{self, Conv3dOutputShapeArgs};

fn run(args: &Conv3dOutputShapeArgs, input: &[f32], kernel: &[f32]) -> Vec<f32> {
    assert!(conv3d::conv3d_is_compatible(args));
    let shape = conv3d::conv3d_output_shape(args);
    let mut out = vec![0.0f32; shape.iter().product()];
    conv3d::conv3d_forward(args, input, kernel, &mut out);
    out
}
```

## Performance levers

- Call `conv3d_is_compatible` once at setup; the forward pass then trusts the
  layout and skips per-call validation.
- Preallocate the output from `conv3d_output_shape` — no reallocation.

## Integration

Configured by [conv_net](../training/conv_net.md); shares the compatibility
philosophy of [conv5d](../geometry/conv5d.md).
