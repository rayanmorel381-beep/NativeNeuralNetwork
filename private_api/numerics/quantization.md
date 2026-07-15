# numerics/quantization

Weight quantization: symmetric int8 and ternary, with matvec kernels that run
directly on the compressed form.

Façade: `private_api::modules::quantization`.

## Key types

- `QuantError` — failure reason.

## Functions

- `quantize_i8_symmetric(...)` — f32 → symmetric int8 with a scale.
- `quantize_ternary(...)` / `ternary_packed_len(...)` — f32 → {-1,0,+1}, packed.
- `matvec_w8(...)` — matrix·vector with int8 weights.
- `matvec_ternary(...)` — matrix·vector with ternary weights.

## Concrete usage

Quantize a weight matrix to int8 once, then run inference matvecs on the
compressed weights (~4× smaller, no dequant needed):

```rust
use native_neural_network::private_api::modules::quantization;

fn quantize(weights_f32: &[f32], q_out: &mut [i8]) -> f32 {
    quantization::quantize_i8_symmetric(weights_f32, q_out)  // returns the scale
}

fn linear_i8(qw: &[i8], scale: f32, x: &[f32], y: &mut [f32]) {
    quantization::matvec_w8(qw, scale, x, y);   // y = (scale * qw) · x
}
```

Ternary packs even tighter:

```rust
let packed_len = quantization::ternary_packed_len(weights_f32.len());
let mut packed = vec![0u8; packed_len];
quantization::quantize_ternary(weights_f32, &mut packed);
```

## Performance levers

- int8 cuts memory ~4× with small accuracy loss; ternary cuts far more for
  tolerant layers.
- `matvec_w8` / `matvec_ternary` operate on packed weights — avoid dequantizing
  back to f32 in the hot path.
- Keep the quantized model a valid container via
  [model_format](../core/model_format.md).

## Integration

Produces the [tensor](../core/tensor.md) `QuantizedTensor`/`PackedTensor` and
[lm](../transformer/lm.md) `QWeight`/`TWeight` forms; matvecs feed
[inference](../transformer/inference.md).
