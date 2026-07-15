# transformer/attention

Scaled dot-product attention (SDPA) with causal and sparse masking, plus a
numerically stable softmax.

Façade: `private_api::modules::attention`.

## Key types

- `AttentionShape` — head/sequence dimensions (`validate`, `output_len`,
  `score_len`, `key_span`, `value_span`).
- `AttentionMask` — mask description (`dense`, `strided`).
- `StridedHead` — per-head strided layout.
- `AttentionError` — failure reason.

## Functions

- `scaled_dot_product_attention(...)` — the dense SDPA kernel.
- `sdpa_strided(...)` — strided-layout SDPA.
- `sdpa_with_sparse_mask(...)` — SDPA with a sparse mask.
- `apply_causal_mask_row(...)` / `apply_sparse_mask_row(...)` — per-row masking.
- `stable_softmax_row(...)` — max-subtracted softmax to avoid overflow.

## Concrete usage

Compute one attention row for autoregressive decoding — mask future positions,
then softmax stably:

```rust
use native_neural_network::private_api::modules::attention;

// `scores` holds Q·Kᵀ for the current query position, length = context so far.
fn attend_row(scores: &mut [f32], query_pos: usize) {
    attention::apply_causal_mask_row(scores, query_pos); // hide future tokens
    attention::stable_softmax_row(scores);               // overflow-safe softmax
    // `scores` are now attention weights to combine with V.
}
```

## Performance levers

- Use `sdpa_strided` to attend over a [kv_cache](../transformer/kv_cache.md)
  without copying keys/values into a contiguous buffer.
- `apply_causal_mask_row` keeps generation O(context) per new token.
- Sparse masks cut work when most positions are masked out.
- Always route scores through `stable_softmax_row` for long contexts.

## Integration

Called by [inference](../transformer/inference.md) transformer blocks; positions
come from [rope](../transformer/rope.md), cached state from
[kv_cache](../transformer/kv_cache.md).
