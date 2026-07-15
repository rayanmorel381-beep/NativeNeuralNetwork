# core/model_format

Encode/decode of the RMD1 container — the on-disk `.rnn` file. **This is the
single source of truth for `.rnn` validity.** Never build a container by hand;
always go through this module so headers, payloads and crypto stay consistent.

Façade: `private_api::modules::model_format`.

## Key types

- `Header` — fixed-size RMD1 header (`RMD1_HEADER_SIZE`).
- `ReadableRnnFormat` — parsed, validated view of a container.
- `ContainerBuildRequest` — inputs needed to assemble a container.
- `RuntimeInput`, `BlobWrite` — payload description and writer.
- `ModelFormatError` — parse/validation failure.

## Functions

- `parse_header(...)` / `write_header(...)` — header round-trip.
- `is_header_consistent(...)` — structural validation.
- `read_rnn_format(...)` — full parse (backs the public `read_rnn`).
- `encode_model(...)` / `encoded_size(...)` /
  `expected_encoded_size_from_header(...)` — sizing and encoding.
- `build_container_from_rmd1[...]` variants — assemble with optional benchmark,
  conv spec, or runtime input.
- `fill_blob_payloads(...)`, `has_full_payload(...)`,
  `build_default_benchmark_blob(...)`, `validate_benchmark_flags(...)`.
- `scalar_bytes_for_dtype(...)` — dtype size helper.

## Concrete usage

Validate an `.rnn` blob's header before trusting any payload — reject early on a
malformed or truncated file:

```rust
use native_neural_network::private_api::modules::model_format;

fn validate(bytes: &[u8]) -> bool {
    match model_format::parse_header(bytes) {
        Ok(header) => model_format::is_header_consistent(&header)
            && model_format::has_full_payload(&header, bytes.len()),
        Err(_) => false,
    }
}
```

Prefer the public [`read_rnn`](../README.md) for the full validated view; use the
low-level helpers only when you need header-only inspection.

## Performance levers

- Call `expected_encoded_size_from_header` to preallocate the exact output
  buffer, avoiding reallocation.
- Use `has_full_payload` to distinguish a metadata-only header from a full model
  before reading large blobs.

## Integration

Reads/writes what [network](../core/network.md), [lm](../transformer/lm.md),
[benchmark](../runtime/benchmark.md) and [crypto](../runtime/crypto.md) produce.
Backs the public `read_rnn` entry point.
