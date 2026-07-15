# core/layers

Layer description, validation and execution planning.

Façade: `private_api::modules::layers`.

## Key types

- `LayerSpec` — declarative description of one layer (sizes, activation).
- `LayerDesc` — resolved descriptor used at build time.
- `LayerPlan` / `LayerPlanF64` — precomputed execution plan (f32/f64).
- `LayerError` — validation failure reason.

## Functions

- `build_from_layers(...)` — turn specs into an executable plan.
- `validate(...)` / `validate_ranges(...)` — check a layer/topology is coherent.
- `layer_chain_is_compatible(...)` — verify adjacent layers connect.
- `input_size()` / `output_size()` / `max_width()` / `chain_widths()` /
  `total_neurons()` — shape queries.
- `weight_len(...)` — parameter count for a layer.
- `layout_spec()` / `desc()` — access the resolved layout/descriptor.

## Concrete usage

Validate a topology once, then size a single scratch arena that covers the widest
layer:

```rust
use native_neural_network::private_api::modules::layers;
use native_neural_network::private_api::modules::scratch::Scratch;

const TOPOLOGY: &[usize] = &[128, 256, 128, 64];

// Reject impossible chains before touching the hot path.
assert!(layers::layer_chain_is_compatible(TOPOLOGY));

// One buffer big enough for the widest activation row.
let widest = layers::max_width(TOPOLOGY);
let mut backing = vec![0.0f32; widest];
let arena = Scratch::new(&mut backing);
```

## Performance levers

- Validate once with `validate_ranges` at build time; the hot path then trusts
  the `LayerPlan` and skips bounds re-checking.
- Use `max_width` / `chain_widths` to size [scratch](../core/scratch.md) arenas.

## Integration

Consumed by [network](../core/network.md) assembly and
[inference](../transformer/inference.md).
