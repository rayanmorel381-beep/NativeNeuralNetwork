# core/rnn_format

The low-level `.rnn` scanner/validator. Where
[model_format](../core/model_format.md) *builds and reads* the container,
`rnn_format` is the **parsing/validation engine** underneath it: it walks the
raw bytes, indexes blobs, and reports structural problems.

Path: `native_neural_network::rnn_format` (internal; reached through
[model_format](../core/model_format.md) and the public `read_rnn`).

## Key types

- `ScanReport` — the result of scanning a container (blob index, offsets).
- `HeaderSummary` — a compact summary of the parsed header.
- `RnnHandle` — a borrowed handle over parsed bytes (used internally by
  engine/trainer).
- `Error` — parse/validation failure.

## Functions

- `validate(bytes, device_id)` — validate a container end-to-end (this is what
  the trainer calls before touching a model).
- `parse_rnn_from_bytes(...)` — produce a handle for blob lookup.
- `find_blob_index(...)` — locate a payload blob by kind.

## Concrete usage

You normally reach this through the public `read_rnn`. The direct validate call
is what the training path uses as a gate:

```rust
// internal pattern (trainer/entry.rs):
crate::rnn_format::validate(bytes, None)?;   // reject a malformed .rnn early
```

## Contract

- This module is the **structural source of truth** for `.rnn`; do not duplicate
  its parsing elsewhere.
- All public reads flow through it via [model_format](../core/model_format.md) /
  `read_rnn`.

## Integration

Backs [model_format](../core/model_format.md) `read_rnn_format`, feeds
[core/engine](../core/engine.md) `rnn_flow` and the
[trainer](../training/trainer.md) validation gate.
