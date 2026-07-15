# transformer/tokenizer

Byte-level BPE tokenizer: train merges, build a vocab, encode text to ids and
decode back. Vocab and merges serialize to versioned blobs stored in the `.rnn`.

Façade: `private_api::modules::tokenizer`.

## Key types

- `Vocab` — id ↔ token table (`VOCAB_MAGIC`, `VOCAB_VERSION`).
- `MergePair` — a BPE merge rule (`MERGES_MAGIC`, `MERGES_VERSION`).
- `TrainerConfig` / `TrainerScratch` — BPE training configuration/workspace.
- `TokenizerError` — failure reason.

## Functions

- Train: `train_bpe(...)`.
- Encode: `encode_bytes`, `encode_bytes_checked`, `encode_byte`,
  `encode_byte_checked`, `encode_with_bpe`, `apply_bpe_merges`.
- Decode: `decode_ids`, `decode_ids_utf8`, `decode_with_vocab`.
- Lookup: `token_to_id`, `id_to_token`, `unk_id`, `vocab_size`.
- Serialize: `write_vocab_blob`, `parse_vocab_blob`, `write_merges_blob`,
  `parse_merges_blob`.

## Concrete usage

Encode a prompt to ids, then decode a generated id sequence back to UTF-8 text:

```rust
use native_neural_network::private_api::modules::tokenizer::{self, Vocab};

fn round_trip(vocab: &Vocab, merges: &[tokenizer::MergePair], text: &str) -> bool {
    let mut ids = [0u32; 256];
    let n = tokenizer::encode_with_bpe(vocab, merges, text.as_bytes(), &mut ids)
        .unwrap_or(0);

    let mut out = [0u8; 1024];
    let m = tokenizer::decode_ids_utf8(vocab, &ids[..n], &mut out).unwrap_or(0);
    &out[..m] == text.as_bytes()
}
```

## Performance levers

- Train once with `train_bpe`, persist with `write_vocab_blob` /
  `write_merges_blob`, then only encode/decode at runtime.
- Use the `*_checked` encoders on untrusted input to reject invalid bytes.

## Integration

Feeds ids to [embeddings](../transformer/embeddings.md); blobs live inside the
[model_format](../core/model_format.md) container.
