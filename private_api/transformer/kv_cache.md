# transformer/kv_cache

Key/value cache for incremental generation. Instead of recomputing attention
over the whole prompt for every new token, keys and values are appended once and
reused.

Façade: `private_api::modules::kv_cache`.

## Key types

- `KvCacheView` — a view over the cached keys/values.
- `KvCacheError` — failure reason.

## Functions

- `append_token(...)` — add one position's K/V.
- `token_slices(...)` — access cached K/V for attention.
- `token_stride()` — layout stride between tokens.
- `len_tokens()` / `remaining_tokens()` / `is_full()` / `is_empty()` — capacity
  state.
- `clear()` — reset for a new sequence.

## Concrete usage

Guard generation against context overflow and reset between conversations:

```rust
use native_neural_network::private_api::modules::kv_cache::KvCacheView;

fn step(cache: &mut KvCacheView, k: &[f32], v: &[f32]) -> bool {
    if cache.is_full() {
        return false;             // context exhausted — stop generating
    }
    cache.append_token(k, v);     // O(1) append, reused by attention
    true
}

fn new_conversation(cache: &mut KvCacheView) {
    cache.clear();                // drop history, keep the allocation
}
```

## Performance levers

- Size the cache once from [lm](../transformer/lm.md) `kv_cache_f32_count`; never
  grow it mid sequence.
- Check `is_full` / `remaining_tokens` before generation to bound context length.
- Pair with [attention](../transformer/attention.md) `sdpa_strided` so no copy is
  needed.

## Integration

Filled during [inference](../transformer/inference.md) prefill, consumed by
[attention](../transformer/attention.md).
