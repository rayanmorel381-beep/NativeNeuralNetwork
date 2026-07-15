# training/conv_net

Configuration for convolutional training paths, feeding the 3D/5D conv kernels.

Façade: `private_api::modules::conv_net`.

## Key types

- `ConvLayerParams` — kernel/stride/padding and channel counts for a conv layer.
- `ConvTrainCfg` — training configuration for the convolutional path.

## Concrete usage

Describe a conv layer and its training configuration to drive the
[conv3d](../geometry/conv3d.md) / [conv5d](../geometry/conv5d.md) forward passes:

```rust
use native_neural_network::private_api::modules::conv_net::{ConvLayerParams, ConvTrainCfg};

// Populate ConvLayerParams (channels, kernel, stride, padding) and a
// ConvTrainCfg, then hand them to the conv3d/conv5d forward/backward kernels.
fn describe(layer: &ConvLayerParams, cfg: &ConvTrainCfg) {
    let _ = (layer, cfg);
}
```

## Performance levers

- Match `ConvLayerParams` shapes to what
  [conv3d](../geometry/conv3d.md)/[conv5d](../geometry/conv5d.md)
  `*_is_compatible` accept, so the kernel never rejects the layout at runtime.

## Integration

Drives [conv3d](../geometry/conv3d.md) and [conv5d](../geometry/conv5d.md); wired
into the [trainer](../training/trainer.md) when a conv spec is present in the
[model_format](../core/model_format.md) container.
