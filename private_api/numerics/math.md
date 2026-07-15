# numerics/math

`no_std` transcendental math. With zero external dependencies, the crate ships
its own `sin`/`cos`/`exp`/`ln`/`pow`/`sqrt`/`tanh`/`round` for both f32 and f64.

Façade: `private_api::modules::math`.

## Key types

- `Float` — the trait abstracting f32/f64 so numeric code stays precision-generic.

## Functions

f32 / f64 pairs:

- `sinf` / `sind`, `cosf` / `cosd`
- `expf` / `expd`, `lnf` / `lnd`
- `powf` / `powd`, `sqrtf` / `sqrtd`
- `tanhf` / `tanhd`, `roundf` / `roundd`

## Concrete usage

These back every activation and normalization. Use them directly wherever you'd
reach for `std::f32` math (which isn't available under `#![no_std]`):

```rust
use native_neural_network::private_api::modules::math;

let gelu_tanh_arg = 0.7978845608f32 * (x + 0.044715 * math::powf(x, 3.0));
let y = 0.5 * x * (1.0 + math::tanhf(gelu_tanh_arg));
```

Write precision-generic kernels with the `Float` trait:

```rust
use native_neural_network::private_api::modules::math::Float;

fn softplus<T: Float>(x: T) -> T {
    (x.exp() + T::one()).ln()
}
```

## Performance levers

- Pick the exact-precision function (`*f` vs `*d`) to avoid needless promotion.
- Prefer `Float`-generic code so the same kernel serves f32 and f64.

## Integration

Underlies [activations](../numerics/activations.md),
[normalization](../transformer/normalization.md),
[attention](../transformer/attention.md) softmax and
[sampling](../transformer/sampling.md).
