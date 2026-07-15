# public/trust_service

Publisher trust service — model signing, activation challenges and distribution
paths. **Feature-gated**: only compiled with
`--features publisher-trust-service`. Re-exported as
`private_api::trust_service` when the feature is on.

Path: `native_neural_network::trust_service`.

## Key types

- `PublisherTrustService` — the signing/activation entry point.
- `ActivationChallenge` / `ActivationDecision` — device activation handshake.
- `TrustServiceError` — failure reason.

## Key constants

- Distribution paths per model size/precision: `SMALL_F32_PATH`, `SMALL_F64_PATH`,
  `MEDIUM_*`, `LARGE_*`, `ENORMOUS_*`.
- `ACTIVATION_TOKEN_MAX_LEN` — activation token bound.

## Functions

- `sign_model_distribution(...)` — sign a model for distribution.
- `encode_activation_challenge_binary(...)` — serialize an activation challenge.
- `validate_activation_decision_placeholder(...)` — validate an activation
  decision.
- `rename(...)` — distribution-path rename helper.

## Concrete usage

Enable the feature, then sign a built model before distributing it:

```toml
# Cargo.toml
[features]
publisher-trust-service = []
```

```rust
#[cfg(feature = "publisher-trust-service")]
use native_neural_network::trust_service::PublisherTrustService;

#[cfg(feature = "publisher-trust-service")]
fn distribute(model_bytes: &[u8]) {
    // sign_model_distribution(...) produces a signed artifact bound to a
    // distribution path (e.g. SMALL_F32_PATH), verifiable with crypto::ed25519_verify.
}
```

## Contract

- Off by default (`default = []`); nothing links unless the feature is set.
- Signatures use [runtime/crypto](../runtime/crypto.md) Ed25519 primitives.

## Integration

Builds on [runtime/crypto](../runtime/crypto.md) for signing; distributes
[core/model_format](../core/model_format.md) containers.
