# numerics/activations

Activation functions and their derivatives, with output-range validation. All
kinds are serializable to/from a `u8` tag so they survive the `.rnn` round-trip.

Façade: `private_api::modules::activations`.

## Key types

- `ActivationKind` — the activation selector (`from_u8`, `to_u8`).

## Functions

- Apply: `apply`, `apply_f64`, `apply_in_place`, `apply_in_place_f64`, `apply_t`.
- Derivatives: `derivative_from_output`, `derivative_from_output_f64`.
- Guards: `derivative_is_finite`, `derivative_is_finite_f64`,
  `validate_activation_outputs`, `validate_activation_outputs_f64`.

## Concrete usage

Apply an activation in place on the forward pass, and use the cached output to
get the derivative on the backward pass:

```rust
use native_neural_network::private_api::modules::activations::{self, ActivationKind};

fn forward(kind: ActivationKind, pre: &mut [f32]) {
    activations::apply_in_place(kind, pre);          // pre -> post, in place
}

fn backward(kind: ActivationKind, post: &[f32], grad: &mut [f32]) {
    for i in 0..post.len() {
        grad[i] *= activations::derivative_from_output(kind, post[i]);
    }
}
```

Persist a chosen activation as one byte:

```rust
let tag = ActivationKind::to_u8(kind);         // store in the container
let restored = ActivationKind::from_u8(tag);   // reload later
```

## Performance levers

- `derivative_from_output` reuses the forward output — no recomputation of the
  pre-activation.
- Use `validate_activation_outputs` on untrusted paths to catch NaN/Inf early.

## Integration

Selected per [layers](../core/layers.md) `LayerSpec`, evaluated inside
[inference](../transformer/inference.md); built on
[math](../numerics/math.md).
