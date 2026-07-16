# Cargo feature: `publisher-trust-service`

Reference for the optional `publisher-trust-service` Cargo feature of
`native_neural_network`.

> Companion to [private_api.md](private_api.md). This document describes **one Cargo
> feature flag** — what it compiles in, what it adds to the `.rnn` container, and the
> public surface it exposes.

---

## Summary

`publisher-trust-service` is an **opt-in, `no_std`, zero-dependency** feature that adds a
*model distribution trust layer* on top of the `.rnn` format. When enabled it provides:

- **Ed25519 signing** of a `.rnn` container for distribution.
- **Verification** of a distributed model against a device or a set of trusted
  publisher keys.
- Two **distribution policies**: publisher-shared and user-device-locked.
- An **online activation** challenge/decision contract (transport-agnostic).
- **Non-destructive `.rnn` update**: migrate an existing model into the unified LMLP
  container (no longer split between dense/MLP and LM), preserving existing blobs.
- **Mutual cross-training**: neighboring trained models act as teacher/student and train
  one another (warm-start + distillation) through a local model pool.

It is **off by default**. When disabled, none of the above is compiled and the crate,
its public API, and the `.rnn` format behave exactly as without the feature (zero cost).

---

## Declaration

From [Cargo.toml](Cargo.toml):

```toml
[features]
default = []
ffi-cdylib = []
publisher-trust-service = []
```

It carries no dependencies and pulls in no other feature.

## Enabling it

Build or test the crate with the flag:

```bash
cargo build   --features publisher-trust-service
cargo test    --features publisher-trust-service
```

As a dependency:

```toml
native_neural_network = { version = "0.4", features = ["publisher-trust-service"] }
```

---

## What the feature changes

| Area | Without the feature | With the feature |
|------|---------------------|------------------|
| Module | `security::trust_service` not compiled | compiled and re-exported at the crate root (`pub use trust_service::*;`) |
| `.rnn` container | dense/LM blobs only | four additional `auth.*` blobs recognized |
| Validation | standard contract checks | adds the `Distribution` validation path (signature/policy enforcement) |
| `.rnn` update | read-only detection of legacy/LMLP containers, with an advisory message | in-place, non-destructive migration to the unified LMLP container |
| Training | standard training | adds mutual cross-training (teacher/student) between neighboring models |
| Public surface | five entry points + facade | plus the signing and activation API below |

The gate is declared in [src/lib.rs](src/lib.rs):

```rust
#[cfg(feature = "publisher-trust-service")]
pub use security::trust_service;
#[cfg(feature = "publisher-trust-service")]
pub use trust_service::*;
```

---

## Distribution policies

A signed model carries a one-byte policy (constants in
[src/format/model_config/ingest.rs](src/format/model_config/ingest.rs)):

| Constant | Value | Meaning |
|----------|-------|---------|
| `DISTRIBUTION_POLICY_PUBLISHER_SHARED` | `1` | Published broadly; verified against a caller-supplied set of **trusted publisher public keys**. The signature does not bind a device. |
| `DISTRIBUTION_POLICY_USER_DEVICE_LOCKED` | `2` | Bound to a specific **device id**; the signature covers that device id, so the model does not validate on any other device. |

## Auth blobs added to the container

Feature-gated TLV blob names (crate-internal constants):

| Blob name | Constant | Content |
|-----------|----------|---------|
| `auth.distribution_policy` | `AUTH_DISTRIBUTION_POLICY_BLOB_DATA` | 1 byte — the policy above |
| `auth.ed25519_sig` | `AUTH_ED25519_SIG_BLOB_DATA` | Ed25519 signature of the canonical message |
| `auth.ed25519_pubkey` | `AUTH_ED25519_PUBKEY_BLOB_DATA` | signer public key |
| `auth.hmac_sha256` | `AUTH_HMAC_SHA256_BLOB_DATA` | reserved HMAC slot — **excluded** from the signed message |

The three `auth.ed25519_*` / `auth.hmac_sha256` blobs are **not** covered by the
signature (they are skipped when building the canonical message), which lets them be
appended after signing.

---

## Signing a distribution

```rust
pub fn sign_model_distribution(
    container: &[u8],
    secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    device_id: &[u8],
    distribution_policy: u8,
    out: &mut [u8],
) -> Result<usize, TrustServiceError>
```

Steps performed ([src/security/trust_service/mod.rs](src/security/trust_service/mod.rs)):

1. Parse the plaintext `.rnn` container.
2. Build a **canonical, length-prefixed message**:
   - `u16` device length + device bytes — the `device_id` for *device-locked*, empty for
     *publisher-shared*;
   - for every model blob (skipping the `auth.*` sig/pubkey/hmac blobs):
     `u16` name length + name + `dtype` + `ndim` + dims + `u64` payload length + payload.
3. **Ed25519-sign** the canonical message with `secret_key`.
4. Append the `auth.distribution_policy`, `auth.ed25519_sig`, and `auth.ed25519_pubkey`
   blobs, and re-assemble the container (with a distribution benchmark blob).
5. Return the number of bytes written into `out`.

Bounds are fixed and checked (no allocation): up to `32` blobs, canonical message capped
at `256 KiB`, `16 KiB` parse scratch.

---

## Verifying a distribution

Verification is reached through the crate's validation subsystem (the
`ValidateRequest::Distribution` variant), which calls the internal
`validate_distribution_secure(bytes, current_device_id, trusted_publisher_pubkeys)` in
[src/format/model_config/validation.rs](src/format/model_config/validation.rs):

1. Run the standard engine/model contract checks.
2. Read `auth.distribution_policy` and branch:
   - **`PUBLISHER_SHARED`** — the trusted-key set must be non-empty; verify the Ed25519
     signature (device = empty); the embedded public key must be present in the trusted
     set, compared in **constant time**.
   - **`USER_DEVICE_LOCKED`** — the current device id must be non-empty; verify the
     Ed25519 signature that binds it.
3. The canonical message is reconstructed exactly as during signing, then verified.

Any mismatch (wrong device, untrusted publisher, tampered blob) fails validation.

---

## Non-destructive `.rnn` update

With the feature enabled, an existing `.rnn` can be **updated into the unified LMLP
container** — the format that no longer separates dense/MLP and LM models into two
distinct on-disk shapes. The update is **non-destructive**: existing blobs (weights,
biases, layer meta, tokenizer, …) are carried over unchanged and re-assembled into the
new container, so no trained data is lost.

> **Status: in progress.** This migration path was started recently and is only partially
> implemented; treat it as work in progress rather than a stable entry point.

## Behavior without the feature (backward compatibility)

Without the feature, the crate stays **read-compatible** with both container generations:

- it **detects** the two container formats / older versions, and
- it **detects when the LM part is absent**.

In that case the model still loads (backward compatibility is preserved) and an
**advisory message** is surfaced, recommending a training run with a dataset so the
`.rnn` becomes complete.

## Mutual cross-training

When the feature is enabled, training can perform **mutual cross-training**: neighboring
already-trained models act as **teacher and student** and train one another, combining a
**warm-start** from a peer model with **distillation** against peer targets. Peers are
exchanged through a **local model pool** on disk, so several models in the same
environment converge together instead of each training in isolation.

This path is gated behind the feature and lives in the training engine
([src/engine/train/trainer/lm_train.rs](src/engine/train/trainer/lm_train.rs)).

---

## Online activation protocol

The feature also defines a **transport-agnostic** activation contract. The crate encodes
challenges and validates decisions; it performs **no network I/O** (it is `no_std`) — the
transport is provided by the caller through the trait.

```rust
pub const ACTIVATION_TOKEN_MAX_LEN: usize = 256;

pub struct ActivationChallenge<'a> {
    pub model_sha256: [u8; 32],
    pub model_signing_pubkey: [u8; ASYM_PUBLIC_KEY_LEN],
    pub distribution_policy: u8,
    pub machine_fingerprint: &'a [u8],
    pub device_id: &'a [u8],
    pub issued_at_unix: u64,
}

pub struct ActivationDecision {
    pub allowed: bool,
    pub expires_unix: u64,
    pub activation_token: [u8; ACTIVATION_TOKEN_MAX_LEN],
    pub activation_token_len: usize,
}

pub trait PublisherTrustService {
    fn request_activation(
        &self,
        challenge: &ActivationChallenge<'_>,
    ) -> Result<ActivationDecision, TrustServiceError>;
}
```

Helpers:

- `encode_activation_challenge_binary(&challenge, out) -> Result<usize, TrustServiceError>`
  — serializes a challenge to a binary frame (magic `ATS0`, version `1`).
- `validate_activation_decision_placeholder(&decision, now_unix) -> Result<(), TrustServiceError>`
  — checks that the decision is `allowed`, the token length is within `(0, 256]`, and the
  token has not expired.

## Errors

```rust
pub enum TrustServiceError {
    InvalidChallenge,
    ServiceUnavailable,
    TransportNotConfigured,
    Unauthorized,
    InvalidDecision,
    TokenExpired,
}
```

`ServiceUnavailable` / `TransportNotConfigured` are part of the contract for caller-side
`PublisherTrustService` implementations.

---

## Public surface (when enabled)

Re-exported at the crate root via `trust_service::*`:

- `sign_model_distribution`
- `encode_activation_challenge_binary`
- `validate_activation_decision_placeholder`
- `ACTIVATION_TOKEN_MAX_LEN`
- `TrustServiceError`
- `ActivationChallenge`, `ActivationDecision`
- `PublisherTrustService`
- `trained` (known `.rnn` paths + an in-place model `rename` helper)

The verification entry point (`validate_distribution_secure`) is crate-internal and is
exercised through the normal validation path, not called directly.

---

## Security properties

- **Ed25519** signatures over a **canonical length-prefixed** serialization, independent
  of TLV ordering; the `auth.*` signature/pubkey/hmac blobs are excluded from the signed
  bytes.
- **Constant-time** comparison when matching trusted publisher keys.
- Fixed-capacity **stack buffers** with explicit bounds checks; `no_std`, **no dynamic
  allocation**.
- Device binding blocks cross-device reuse (`USER_DEVICE_LOCKED`); trusted-key pinning
  blocks unknown publishers (`PUBLISHER_SHARED`).

---

## Interactions

- Independent from `ffi-cdylib`; the two features can be combined or used separately.
- `default = []`, so the trust layer is strictly opt-in and never affects default builds.

## Source references

- [src/security/trust_service/mod.rs](src/security/trust_service/mod.rs) — signing,
  activation types, encoding.
- [src/security/mod.rs](src/security/mod.rs) — module gate.
- [src/format/model_config/ingest.rs](src/format/model_config/ingest.rs) — auth blob
  names and policy constants.
- [src/format/model_config/validation.rs](src/format/model_config/validation.rs) —
  distribution verification path.
- [src/engine/train/trainer/lm_train.rs](src/engine/train/trainer/lm_train.rs) — mutual
  cross-training (warm-start, distillation, local model pool).
- [src/lib.rs](src/lib.rs) — feature-gated re-exports.
