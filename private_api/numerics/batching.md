# numerics/batching

Sequence batching helpers: pad variable-length sequences, build padding masks and
iterate over real (non-pad) tokens.

Façade: `private_api::modules::batching`.

## Key types

- `BatchError` — failure reason.

## Functions

- `pad_sequences_u32(...)` — pad a batch of token-id sequences to equal length.
- `make_padding_mask(...)` — mask marking real vs pad positions.
- `sequence_lengths(...)` / `max_sequence_len(...)` / `count_non_pad(...)` —
  length queries.
- `for_each_token_row(...)` — iterate over token rows, skipping padding.

## Concrete usage

Pad a ragged batch, build its mask, and count the tokens that actually
contribute to the loss:

```rust
use native_neural_network::private_api::modules::batching;

fn prepare(seqs: &[&[u32]], pad_id: u32, out: &mut [u32], mask: &mut [bool]) -> usize {
    let max_len = batching::max_sequence_len(seqs);
    batching::pad_sequences_u32(seqs, pad_id, max_len, out);
    batching::make_padding_mask(out, pad_id, mask);
    batching::count_non_pad(mask)          // real token count for loss scaling
}
```

## Performance levers

- Feed the padding mask into [attention](../transformer/attention.md) so padded
  positions never contribute to scores.
- Scale the loss by `count_non_pad`, not the padded length, so padding doesn't
  bias the gradient.

## Integration

Prepares inputs for [trainer](../training/trainer.md) batched steps and
[inference](../transformer/inference.md); masks feed
[attention](../transformer/attention.md).
