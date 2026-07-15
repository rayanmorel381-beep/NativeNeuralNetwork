# training/lora

Low-Rank Adaptation (LoRA): adapt a frozen base model with a small low-rank
delta instead of retraining all weights.

Façade: `private_api::modules::lora`.

## Key types

- `LoraError` — failure reason.

## Functions

- `lora_forward_delta(...)` — compute the low-rank delta contribution `B·A·x`.
- `apply_lora_delta(...)` — add the delta into the base output.

## Concrete usage

Run a base linear layer, then fold in the LoRA delta — the base weights stay
frozen:

```rust
use native_neural_network::private_api::modules::lora;

// `a`,`b` are the low-rank factors (rank r); `x` the input, `y` the base output.
fn adapted(a: &[f32], b: &[f32], x: &[f32], y: &mut [f32], scale: f32) {
    let mut delta = vec![0.0f32; y.len()];
    lora::lora_forward_delta(a, b, x, &mut delta);   // B·A·x
    lora::apply_lora_delta(y, &delta, scale);        // y += scale * delta
}
```

## Performance levers

- Rank `r` is the cost knob: memory/compute scale with `r`, not the full weight
  matrix — adaptation is far cheaper than full fine-tuning.
- Keep the base model quantized ([quantization](../numerics/quantization.md)) and
  only the LoRA factors in f32.

## Integration

Used at runtime by [inference](../transformer/inference.md) (`generate_lora`,
`LoraModel`) and stored via [lm](../transformer/lm.md) `LmBlockLora` /
`AdaptedWeightBuf`.
