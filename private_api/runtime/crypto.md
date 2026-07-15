# runtime/crypto

Cryptographic primitives for `.rnn` integrity and confidentiality, all
hand-rolled `no_std`: SHA-256/512, HMAC-SHA256, Ed25519 signatures, and
authenticated encryption of the container payload (optionally owner-bound).

Façade: `private_api::modules::crypto`.

## Key types

- `Sha256Ctx` / `Sha512Ctx` — streaming hash contexts.
- `RnnEncryptionMetadata` — the encryption header stored in the container.
- `ConstantTimeEq` — constant-time comparison trait.
- `SignatureError` / `DigestToHexError` — failure reasons.

## Key constants

- Ed25519 sizes: `ASYM_PUBLIC_KEY_LEN`, `ASYM_SECRET_KEY_LEN`,
  `ASYM_SIGNATURE_LEN`.
- Encrypted-container layout: `RNN_ENCRYPTED_*` sizes/offsets,
  `RNN_MIN_ACCEPTED_KEY_VERSION`.

## Representative functions

- Hash/MAC: `compute_hmac_sha256`, `digest_to_hex_lower`.
- Signatures: `ed25519_public_from_secret`, `ed25519_sign`, `ed25519_verify`.
- Encryption: `encrypt_rnn`, `decrypt_rnn`, `encrypt_payload`, `decrypt_payload`,
  and `*_owner_bound` variants for device/owner binding.
- Compare: `constant_time_eq`.

## Concrete usage

Verify a container's Ed25519 signature before trusting its contents, using a
constant-time comparison to avoid timing leaks:

```rust
use native_neural_network::private_api::modules::crypto;

fn verify(public_key: &[u8], message: &[u8], signature: &[u8]) -> bool {
    crypto::ed25519_verify(public_key, message, signature)
}

fn tags_equal(a: &[u8], b: &[u8]) -> bool {
    crypto::constant_time_eq(a, b)   // no early-exit timing side channel
}
```

Owner-bound decryption ties a model to a specific device id:

```rust
// decrypt_rnn_payload_owner_bound(bytes, key, device_id) -> plaintext
```

## Performance levers

- Always compare digests/tags with `constant_time_eq` — never `==`.
- Prefer the `*_owner_bound` paths when a model must not run on another device.
- Reject anything below `RNN_MIN_ACCEPTED_KEY_VERSION` to enforce key rotation.

## Integration

Protects the [model_format](../core/model_format.md) container; `read_rnn`'s
optional `device_id` feeds the owner-bound decryption path.
