# numerics/precision

The global precision switch: choose whether compute runs in f32 or f64.

Façade: `private_api::modules::precision`.

## Key types

- `Precision` — the f32/f64 selector.

## Functions

- `set_precision(...)` — set the active precision.
- `get_precision()` — read it back.

## Concrete usage

Pick precision once at startup and keep the whole pipeline consistent:

```rust
use native_neural_network::private_api::modules::precision::{self, Precision};

fn configure(double: bool) {
    precision::set_precision(if double { Precision::F64 } else { Precision::F32 });
}

fn is_double() -> bool {
    precision::get_precision() == Precision::F64
}
```

## Performance levers

- f32 halves memory bandwidth and roughly doubles throughput versus f64 for the
  same topology — prefer it unless numerical stability demands f64.
- Set precision **before** building; mixing f32 weights with f64 kernels is a bug.

## Integration

Governs which code path [inference](../transformer/inference.md),
[trainer](../training/trainer.md) and [losses](../training/losses.md) take
(`*_f32` vs `*_f64`). Mirrors [model_config](../core/model_config.md) `Precision`.
