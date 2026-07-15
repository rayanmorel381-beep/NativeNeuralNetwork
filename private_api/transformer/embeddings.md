# transformer/embeddings

Token embedding lookup and the MLP output head.

Façade: `private_api::modules::embeddings`.

## Key types

- `EmbeddingError` — failure reason.

## Functions

- `gather_embeddings(...)` — gather rows from the embedding table for a batch of
  token ids.
- `mlp_head_forward(...)` — project hidden states to output logits.

## Concrete usage

Turn token ids into embedding vectors at the input, and hidden states into logits
at the output:

```rust
use native_neural_network::private_api::modules::embeddings;

fn embed(table: &[f32], ids: &[u32], d_model: usize, out: &mut [f32]) {
    // out.len() == ids.len() * d_model
    embeddings::gather_embeddings(table, ids, d_model, out);
}

fn logits(hidden: &[f32], head_w: &[f32], out: &mut [f32]) {
    embeddings::mlp_head_forward(hidden, head_w, out);
}
```

## Performance levers

- `gather_embeddings` is a pure index-gather; keep token ids contiguous for
  cache-friendly reads.
- The MLP head dominates output cost for large vocabularies — quantize it via
  [quantization](../numerics/quantization.md) / [lm](../transformer/lm.md)
  `QWeight`.

## Integration

Front and back of the transformer stack in
[inference](../transformer/inference.md); the head pairs with
[sampling](../transformer/sampling.md).
