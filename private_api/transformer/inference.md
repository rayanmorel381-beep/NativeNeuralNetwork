# transformer/inference

Forward/backward passes and text generation, including runtime LoRA. This is
what the public `run` entry point drives.

Façade: `private_api::modules::inference`.

## Key types

- `ForwardScratch` / `BackwardScratch` / `TrainActivations` / `BlockActivations`
  — reusable working buffers (size via `f32_count`, `total_f32_count`,
  `prefill_buf_count`).
- `BlockQuant` — quantized block state.
- `GenerateArgs` / `GenerateOutcome` — generation request/result.
- `LmRunInputs` / `LmRunBufs`, `LmTextInputs` / `LmTextBufs`,
  `LmChatInputs` / `LmChatBufs` / `LmChatLearn` — run/text/chat I/O bundles.
- `LoraModel` / `LoraSpec` / `LoraWeights` — runtime LoRA.
- `TokenSink` — streaming token output.
- `InferenceError` — failure reason.

## Representative functions

- Generation: `generate`, `generate_beam`, `generate_lora`, `run_lm`,
  `run_lm_text`, `run_lm_chat`, `prefill_prompt`.
- Transformer: `transformer_forward`, `transformer_block_forward`,
  `transformer_block_backward`, `mlp_head_backward`.
- Numerics: `softmax_stable`, `normalize_logits_in_place`, `silu_prime`,
  `argmax_index`.
- State: `next_position`, `completed_early`.

## Concrete usage

The simplest way to generate is the public `run`; internally it flows
prompt → prefill → per-token decode. Stream tokens through a `TokenSink`:

```rust
use native_neural_network::private_api::modules::inference;

// Pseudo-flow inside a generation loop:
// 1) prefill the prompt once (fills the kv-cache)
// 2) decode token-by-token, reusing ForwardScratch
fn decode_step(
    scratch: &mut inference::ForwardScratch,
    logits: &mut [f32],
) -> usize {
    inference::normalize_logits_in_place(logits);
    inference::argmax_index(logits)          // greedy pick
}
```

For chat, use `run_lm_chat` with `LmChatInputs`; for plain completion,
`run_lm_text` with `LmTextInputs`.

## Performance levers

- Reuse `ForwardScratch`/`BackwardScratch` across steps; never reallocate per
  token.
- Use `prefill_prompt` + [kv_cache](../transformer/kv_cache.md) so subsequent
  tokens only run incremental attention.
- `generate_beam` trades throughput for quality
  (see [beam_search](../transformer/beam_search.md)); plain `generate` with
  [sampling](../transformer/sampling.md) is faster.

## Integration

Runs [lm](../transformer/lm.md) weights over
[attention](../transformer/attention.md), [rope](../transformer/rope.md),
[normalization](../transformer/normalization.md), feeding
[sampling](../transformer/sampling.md).
