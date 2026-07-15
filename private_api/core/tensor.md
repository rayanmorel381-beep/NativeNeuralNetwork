# core/tensor

The tensor families used across the crate. Each type trades memory, speed and
mutability differently, so pick the one that matches the access pattern.

Façade: `private_api::modules::tensor`.

## Tensor types

- `DenseTensor` — contiguous, fully materialized values.
- `StaticTensor` — fixed, compile-time-sized storage (no allocation).
- `TrainingTensor` — carries values plus gradient/accumulation state.
- `QuantizedTensor` / `PackedTensor` — int8/ternary compressed storage.
- `SparseTensor` — stores only non-zero entries.
- `LazyTensor` — deferred/evaluated on demand (`evaluate`).
- `CacheTensor` — cached intermediate.
- `SharedTensor` — shared backing (e.g. across fork workers).
- `GpuTensor` — device-side buffer.
- `TensorView` / `TensorViewMut` / `TensorViewRo` — borrows without copy.

## Representative functions

- Construction: `new`, `alloc`, `from_raw`, `from_slice`, `from_slice_offset`,
  `from_slices`, `from_qweight`.
- Access: `as_slice`, `as_mut_slice`, `as_ptr`, `as_mut_ptr`, `as_bytes`,
  `as_bytes_mut`, `get`, `get_mut`, `idx_linear`, `numel`, `len`, `is_empty`.
- Compute: `axpy`, `apply_grad`, `accumulate_grad`, `apply_mask`, `fill`,
  `fill_zero`, `checksum`, `dequant_row`, `dequant_into`.
- Validation: `is_valid_layout`, `numel_f32`.

## Concrete usage

Wrap an existing slice as a dense tensor and run a fused `y += a*x` (axpy) with
no allocation:

```rust
use native_neural_network::private_api::modules::tensor::DenseTensor;

let x = [1.0f32, 2.0, 3.0, 4.0];
let mut y = [10.0f32, 10.0, 10.0, 10.0];

let xt = DenseTensor::from_slice(&x);
let mut yt = DenseTensor::from_slice(&mut y);
yt.axpy(0.5, &xt);          // y = y + 0.5*x
assert_eq!(y, [10.5, 11.0, 11.5, 12.0]);
```

Shrink weights ~4× by dequantizing an int8 row only when needed:

```rust
use native_neural_network::private_api::modules::tensor::QuantizedTensor;

fn read_row(q: &QuantizedTensor, row: usize, out: &mut [f32]) {
    q.dequant_into(row, out);   // materialize a single row on demand
}
```

## Performance levers

- Prefer `TensorView*` borrows over copies in hot loops.
- Use `QuantizedTensor`/`PackedTensor` to cut memory ~4× (int8) or more (ternary);
  pair with [quantization](../numerics/quantization.md) matvec kernels.
- `SharedTensor` is the right choice for parameters accessed by the
  [trainer](../training/trainer.md) fork pool.
- `LazyTensor` avoids materializing intermediates you may never read.

## Integration

Storage layer under [inference](../transformer/inference.md),
[trainer](../training/trainer.md), [engine](../core/engine.md) and
[quantization](../numerics/quantization.md).
