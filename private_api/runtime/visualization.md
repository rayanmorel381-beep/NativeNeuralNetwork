# runtime/visualization

Read-only views over a network for inspection and rendering: walk layers,
neurons and weights, and fill a mesh buffer for 3D/5D visualization.

Façade: `private_api::modules::visualization`.

## Key types

- `NetworkView` — top-level view (`layer_count`, `input_count`, `header_size`).
- `LayerView` — a layer (`activation`, `neuron_count`, `weight_count`,
  `biases_offset`, `layer_meta_size`).
- `NeuronView` — a neuron (`weight_at`, `weight_count`, `weight_bytes`).
- `VisualizeError` — failure reason.

## Functions

- `get_network_view(...)` — parse a view from container bytes.
- `layer(...)` / `neuron(...)` — descend into layers/neurons.
- `fill_mesh_from_bytes(...)` / `mesh_required_buffers_from_bytes(...)` — build a
  render mesh and size its buffers.

## Concrete usage

Walk a trained model's structure straight from the container bytes, then size and
fill a mesh for rendering:

```rust
use native_neural_network::private_api::modules::visualization as viz;

fn inspect(rnn_bytes: &[u8]) {
    let net = viz::get_network_view(rnn_bytes).expect("bad container");
    for l in 0..net.layer_count() {
        let layer = viz::layer(&net, l);
        let _ = (layer.neuron_count(), layer.weight_count(), layer.activation());
    }

    let (verts, indices) = viz::mesh_required_buffers_from_bytes(rnn_bytes);
    let mut vbuf = vec![0.0f32; verts];
    let mut ibuf = vec![0u32; indices];
    viz::fill_mesh_from_bytes(rnn_bytes, &mut vbuf, &mut ibuf);
}
```

## Performance levers

- Views are zero-copy over the container bytes — no model deserialization.
- Call `mesh_required_buffers_from_bytes` first to allocate mesh buffers exactly
  once.

## Integration

Reads the [model_format](../core/model_format.md) container; geometry pairs with
[sphere5d](../geometry/sphere5d.md).
