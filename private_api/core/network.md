# core/network

The in-memory neural network structure and its assembly helpers.

Façade: `private_api::modules::network`.

## Key types

- `NeuralNetwork` — the model: layer chain plus weights/biases.
- `NetworkStats` — summary counts (layers, neurons, parameters).

## Functions

- `from_parts(...)` — assemble a network from already-laid-out parts.
- `build_layer_specs(...)` — derive per-layer specs from a description.
- `layer_count()` — number of layers.
- `expected_weights_count(...)` / `expected_biases_count(...)` — validate that a
  buffer holds exactly the parameters a topology requires.
- `network_stats()` — produce a `NetworkStats`.
- `conceptualize_5d(...)` — project the network into the 5D representation used
  by [sphere5d](../geometry/sphere5d.md).

## Concrete usage

Size buffers *exactly* for a topology before allocating — no reallocation, which
is how the training binaries pin their static weight/bias arrays:

```rust
use native_neural_network::private_api::modules::network;

const TOPOLOGY: &[usize] = &[128, 256, 128, 64];

let n_weights = network::expected_weights_count(TOPOLOGY);
let n_biases = network::expected_biases_count(TOPOLOGY);

let mut weights = vec![0.0f32; n_weights];
let mut biases = vec![0.0f32; n_biases];
// weights/biases are now guaranteed to fit build_f32's requirements.
```

## Performance levers

- Call the `expected_*_count` helpers before allocating so you size buffers once.
- Reuse a single `NeuralNetwork` across inference calls; pair it with
  [scratch](../core/scratch.md) buffers to avoid per-call allocation.

## Integration

Built from [layers](../core/layers.md) specs and
[initializers](../training/initializers.md), stored in the
[model_format](../core/model_format.md) container, executed via
[inference](../transformer/inference.md).
