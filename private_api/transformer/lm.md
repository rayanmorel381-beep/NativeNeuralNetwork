# transformer/lm

The full transformer language model: weights, config, chat handling and
quantization. This is the largest functional module.

Façade: `private_api::modules::lm`.

## Key types

- `LmConfig` — model hyperparameters (`LM_CONFIG_BLOB_LEN` encoded size).
- `LmWeights` / `LmBlockWeights` / `MoeBlockWeights` — parameter storage.
- `LmWeightsGrads` / `LmBlockGrads` — gradient storage for training.
- `LmQuant` / `BlockQWeights` / `QWeight` / `BlockTWeights` / `TWeight` —
  quantized weight forms.
- `LmBlockLora` / `LoraForward` / `AdaptedWeightBuf` — LoRA adaptation.
- `LmBuildParams` — build-time configuration.
- `LmContainerView` — a parsed model view from a container.
- `MlpHead` — the output projection head.
- `SamplingConfig` — decoding configuration (`default_sampling`, `greedy`).
- `ChatMessage` / `ChatRole` / `ChatSpecialTokens` — chat framing.
- `Lcg` — small deterministic RNG (`jumped`, `base`).
- `LmError` — failure reason.

## Representative functions

- Build: `lm_build`, `lm_build_init`.
- Config: `lm_config_encode`, `lm_config_decode`.
- Container: `lm_container_read`.
- Chat: `build_chat_prompt`, `chat_stop_ids`.
- Sizing: `forward_scratch_f32_count`, `backward_scratch_f32_count`,
  `kv_cache_f32_count`, `block_activations_f32_count`, `int8_i8_count`,
  `int8_scales_f32_count`, `head_dim`, `kv_h`, `kv_head_groups`.

## Concrete usage

`LM_PARAMS` layout used by every training binary (order matters):

```text
[ vocab_size, d_model, d_ff, ctx_len, n_layers, n_heads, n_kv_heads,
  head_dim, ffn_expansion, misc ]
// e.g. small model:
const LM_PARAMS: &[usize] =
    &[32768, 1024, 2048, 4096, 16, 16, 8, 128, 16384, 512];
```

Size all runtime buffers deterministically from the config, then reuse them:

```rust
use native_neural_network::private_api::modules::lm;

fn plan_buffers(cfg: &lm::LmConfig) -> (usize, usize, usize) {
    let fwd = lm::forward_scratch_f32_count(cfg);
    let kv  = lm::kv_cache_f32_count(cfg);
    let act = lm::block_activations_f32_count(cfg);
    (fwd, kv, act)   // allocate each once, reuse across tokens
}
```

## Performance levers

- Use the `*_f32_count` helpers to size a single [scratch](../core/scratch.md)
  arena and the [kv_cache](../transformer/kv_cache.md) exactly.
- Quantize with `LmQuant`/`QWeight` to shrink weights; keep the container valid
  via [model_format](../core/model_format.md).
- Configure `SamplingConfig` (see [sampling](../transformer/sampling.md)) once per
  generation.

## Integration

Executed by [inference](../transformer/inference.md), attention via
[attention](../transformer/attention.md), positions via
[rope](../transformer/rope.md), trained by [trainer](../training/trainer.md).
