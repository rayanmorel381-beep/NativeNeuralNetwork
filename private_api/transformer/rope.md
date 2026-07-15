# transformer/rope

Rotary position embeddings (RoPE). Encodes token positions by rotating query/key
pairs, enabling relative-position awareness without learned position tables.

Façade: `private_api::modules::rope`.

## Key types

- `RopeError` — failure reason.

## Functions

- `precompute_rope_freqs(...)` — build the frequency table once.
- `apply_rope_in_place(...)` — rotate a query/key buffer in place.
- `apply_rope_from_cache(...)` — apply using a precomputed frequency cache.

## Concrete usage

Precompute the frequency table at model build, then rotate each token's Q/K from
the cache — no per-token trigonometry:

```rust
use native_neural_network::private_api::modules::rope;

// Once, at build time:
let mut freqs = vec![0.0f32; head_dim * max_ctx];
rope::precompute_rope_freqs(&mut freqs, head_dim, max_ctx);

// Per token, in the hot path:
fn rotate(freqs: &[f32], q: &mut [f32], k: &mut [f32], pos: usize) {
    rope::apply_rope_from_cache(q, freqs, pos);
    rope::apply_rope_from_cache(k, freqs, pos);
}
```

## Performance levers

- Call `precompute_rope_freqs` once; reuse via `apply_rope_from_cache`.
- In-place application avoids extra buffers on the
  [inference](../transformer/inference.md) hot path.

## Integration

Applied inside [attention](../transformer/attention.md) before the score
computation, using the positions tracked by
[kv_cache](../transformer/kv_cache.md).
