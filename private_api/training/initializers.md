# training/initializers

Deterministic parameter initialization: He/uniform, zeros, constant, for f32 and
f64.

Façade: `private_api::modules::initializers`.

## Key types

- `InitKindF32` / `InitKindF64` — initializer selection per precision.
- `InitError` — failure reason.

## Functions

- `initialize_parameters_f32(...)` / `initialize_parameters_f64(...)` — dispatch
  on the init kind.
- He-uniform: `initialize_parameters_f32_he_uniform`, `..._f64_he_uniform`.
- Zeros: `initialize_parameters_f32_zeros`, `..._f64_zeros`.
- Constant: `initialize_parameters_f32_constant`, `..._f64_constant`.
- Validation: `expected_parameter_counts`, `layers_are_valid`,
  `init_kind_is_finite_f32`, `init_kind_is_finite_f64`.

## Concrete usage

He-initialize weights for a topology with a fixed seed (reproducible builds):

```rust
use native_neural_network::private_api::modules::initializers;

const TOPOLOGY: &[usize] = &[128, 256, 128, 64];

let (n_w, n_b) = initializers::expected_parameter_counts(TOPOLOGY);
let mut weights = vec![0.0f32; n_w];
let mut biases = vec![0.0f32; n_b];

initializers::initialize_parameters_f32_he_uniform(TOPOLOGY, &mut weights, 0xA1B2_C3D4);
initializers::initialize_parameters_f32_zeros(&mut biases);
```

## Performance levers

- Use `expected_parameter_counts` to size buffers exactly (matches
  [network](../core/network.md) counts).
- Seed the He initializer for reproducible experiments.

## Integration

Fills the buffers passed to `build_f32`/`build_f64`; counts align with
[layers](../core/layers.md) / [network](../core/network.md).
