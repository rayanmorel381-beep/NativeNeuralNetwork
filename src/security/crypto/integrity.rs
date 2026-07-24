use crate::security::crypto::{constant_time_eq, sha256_bytes, sha512_bytes, ConstantTimeEq};
use super::auth;
use core::sync::atomic::{AtomicU64, Ordering};

pub fn verify_sha256(data: &[u8], expected: &[u8; 32]) -> bool {
    let mut out = [0u8; 32];
    sha256_bytes(data, &mut out);
    out.ct_eq(expected)
}

pub fn verify_sha512(data: &[u8], expected: &[u8; 64]) -> bool {
    let mut out = [0u8; 64];
    sha512_bytes(data, &mut out);
    out.ct_eq(expected)
}

const RNN_CIPHER_SALT: [u8; 16] = [
    0x9a, 0x3f, 0xb1, 0x07, 0xe2, 0x5c, 0xd4, 0x68,
    0x7b, 0x11, 0xf0, 0x4e, 0xa9, 0x63, 0xc8, 0x2d,
];

fn derive_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    sha256_bytes(&RNN_CIPHER_SALT, &mut key);
    key
}

fn derive_owner_secret(base_key: &[u8; 32]) -> [u8; auth::ASYM_SECRET_KEY_LEN] {
    auth::hmac_sha256(base_key, b"rnn-asymmetric-owner-v1")
}

static ISSUE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn validate_runtime_crypto_material(
    key: &[u8],
    owner_secret_key: &[u8; auth::ASYM_SECRET_KEY_LEN],
    metadata: &auth::RnnEncryptionMetadata,
    header_prefix: &[u8],
) -> bool {
    let metadata_is_supported = metadata.key_version >= auth::RNN_MIN_ACCEPTED_KEY_VERSION
        && metadata.issue_counter != 0
        && metadata.crypto_timestamp != 0
        && metadata.nonce.iter().any(|&byte| byte != 0);
    if !metadata_is_supported {
        return false;
    }

    let mut header_sha = [0u8; 32];
    sha256_bytes(header_prefix, &mut header_sha);
    if !verify_sha256(header_prefix, &header_sha) {
        return false;
    }

    let integrity_key = auth::hmac_sha256(key, b"runtime-integrity");
    let header_digest = auth::compute_hmac_sha256(&integrity_key, header_prefix);
    if !auth::verify_hmac_sha256(&integrity_key, header_prefix, &header_digest) {
        return false;
    }

    let mut header_hex = [0u8; 64];
    let hex_len = crate::security::crypto::digest_to_hex_lower(&header_digest, &mut header_hex).unwrap_or(0);
    if hex_len != 64 {
        return false;
    }

    let owner_public_key = auth::ed25519_public_from_secret(owner_secret_key);
    let signature = auth::sign_message(owner_secret_key, header_prefix);
    let mut signature_buf = [0u8; auth::ASYM_SIGNATURE_LEN];
    signature_buf.copy_from_slice(&signature);
    auth::verify_signature(&owner_public_key, header_prefix, &signature_buf)
}

pub fn encrypt_rnn(data: &mut [u8], plain_len: usize) -> usize {
    if plain_len < 0x20 {
        return plain_len;
    }
    let total = match auth::encrypted_rnn_size(plain_len) {
        Some(t) => t,
        None => return plain_len,
    };
    if data.len() < total {
        return plain_len;
    }

    let base_key = derive_key();
    let owner_secret = derive_owner_secret(&base_key);
    let crypto_timestamp = crate::engine::runtime::hardware::monotonic_ns();
    let issue_counter = ISSUE_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
    let key_version = 1u32;

    let nonce = auth::derive_nonce_v2(
        &base_key, &owner_secret, key_version, issue_counter, crypto_timestamp,
        &data[..plain_len],
    );
    let public_meta = auth::build_public_meta_from_plaintext(&data[..plain_len]);
    let effective_key = auth::derive_owner_bound_key(
        &base_key, &owner_secret, key_version, issue_counter, crypto_timestamp,
    );

    data.copy_within(0..plain_len, auth::RNN_ENCRYPTED_LEN_PREFIX);

    let nonce_start = auth::RNN_ENCRYPTED_MAGIC_SIZE;
    let nonce_end = nonce_start + auth::RNN_ENCRYPTED_NONCE_SIZE;
    let ver_end = nonce_end + auth::RNN_ENCRYPTED_KEY_VERSION_SIZE;
    let ctr_end = ver_end + auth::RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
    let ts_end = ctr_end + auth::RNN_ENCRYPTED_TIMESTAMP_SIZE;

    data[..auth::RNN_ENCRYPTED_MAGIC_SIZE].copy_from_slice(&auth::RNN_ENCRYPTED_MAGIC);
    data[nonce_start..nonce_end].copy_from_slice(&nonce);
    data[nonce_end..ver_end].copy_from_slice(&key_version.to_le_bytes());
    data[ver_end..ctr_end].copy_from_slice(&issue_counter.to_le_bytes());
    data[ctr_end..ts_end].copy_from_slice(&crypto_timestamp.to_le_bytes());
    data[ts_end..auth::RNN_ENCRYPTED_META_END].copy_from_slice(&public_meta);

    let payload_start = auth::RNN_ENCRYPTED_LEN_PREFIX;
    let payload_end = payload_start + plain_len;

    let mut shadow_buf = [0u8; 256 + auth::RNN_ENCRYPTED_HEADER_SIZE + auth::RNN_ENCRYPTED_TAG_SIZE];
    let shadow_len = auth::encrypt_payload(&base_key, &data[..plain_len], &mut shadow_buf)
        .unwrap_or(0);
    if shadow_len != 0 {
        let mut shadow_plain = [0u8; 256];
        let shadow_plain_len = auth::decrypt_payload(&base_key, &shadow_buf[..shadow_len], &mut shadow_plain)
            .unwrap_or(0);
        if shadow_plain_len != plain_len || shadow_plain[..plain_len] != data[..plain_len] {
            return plain_len;
        }
    }

    let metadata = auth::create_encryption_metadata(
        key_version,
        issue_counter,
        crypto_timestamp,
        nonce,
    );
    let header_prefix = &data[..auth::RNN_ENCRYPTED_LEN_PREFIX];
    if !validate_runtime_crypto_material(
        &base_key,
        &owner_secret,
        &metadata,
        header_prefix,
    ) {
        return plain_len;
    }

    let mut digest = [0u8; auth::RNN_ENCRYPTED_DIGEST_SIZE];
    sha512_bytes(&data[payload_start..payload_end], &mut digest);
    data[auth::RNN_ENCRYPTED_META_END..auth::RNN_ENCRYPTED_LEN_PREFIX]
        .copy_from_slice(&digest);

    let stream_key = auth::derive_stream_key_v2(&effective_key, &nonce);
    auth::xor_with_keystream_v2_in_place(
        &stream_key, &nonce, &mut data[payload_start..payload_end],
    );

    let tag = auth::compute_tag_v2(
        &effective_key, &nonce, key_version, issue_counter, crypto_timestamp,
        &data[payload_start..payload_end], plain_len,
    );
    data[payload_end..total].copy_from_slice(&tag);

    if plain_len <= 256 {
        let mut shadow_buf = [0u8; 256 + auth::RNN_ENCRYPTED_HEADER_SIZE + auth::RNN_ENCRYPTED_TAG_SIZE];
        let shadow_len = auth::encrypt_payload_owner_bound(
            &base_key,
            &owner_secret,
            key_version,
            issue_counter,
            crypto_timestamp,
            &data[payload_start..payload_end],
            &mut shadow_buf,
        )
        .unwrap_or(0);
        if shadow_len != 0 {
            let mut shadow_plain = [0u8; 256];
            let shadow_plain_len = auth::decrypt_payload_owner_bound(
                &base_key,
                &owner_secret,
                &shadow_buf[..shadow_len],
                &mut shadow_plain,
            )
            .unwrap_or(0);
            if shadow_plain_len != plain_len
                || shadow_plain[..plain_len] != data[payload_start..payload_end]
            {
                return plain_len;
            }
        }
    }

    total
}

pub fn decrypt_rnn(data: &mut [u8], enc_len: usize) -> usize {
    if enc_len < auth::RNN_ENCRYPTED_HEADER_SIZE {
        return enc_len;
    }
    if !constant_time_eq(
        &data[..auth::RNN_ENCRYPTED_MAGIC_SIZE],
        &auth::RNN_ENCRYPTED_MAGIC,
    ) {
        return enc_len;
    }

    let plain_len = enc_len - auth::RNN_ENCRYPTED_HEADER_SIZE;

    let nonce_start = auth::RNN_ENCRYPTED_MAGIC_SIZE;
    let nonce_end = nonce_start + auth::RNN_ENCRYPTED_NONCE_SIZE;
    let ver_end = nonce_end + auth::RNN_ENCRYPTED_KEY_VERSION_SIZE;
    let ctr_end = ver_end + auth::RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
    let payload_start = auth::RNN_ENCRYPTED_LEN_PREFIX;
    let payload_end = payload_start + plain_len;

    let mut nonce = [0u8; auth::RNN_ENCRYPTED_NONCE_SIZE];
    nonce.copy_from_slice(&data[nonce_start..nonce_end]);

    let key_version = u32::from_le_bytes([
        data[nonce_end], data[nonce_end + 1], data[nonce_end + 2], data[nonce_end + 3],
    ]);
    let issue_counter = u64::from_le_bytes([
        data[ver_end], data[ver_end + 1], data[ver_end + 2], data[ver_end + 3],
        data[ver_end + 4], data[ver_end + 5], data[ver_end + 6], data[ver_end + 7],
    ]);
    let crypto_timestamp = u64::from_le_bytes([
        data[ctr_end], data[ctr_end + 1], data[ctr_end + 2], data[ctr_end + 3],
        data[ctr_end + 4], data[ctr_end + 5], data[ctr_end + 6], data[ctr_end + 7],
    ]);

    if key_version < auth::RNN_MIN_ACCEPTED_KEY_VERSION
        || issue_counter == 0
        || crypto_timestamp == 0
    {
        return enc_len;
    }

    let base_key = derive_key();
    let owner_secret = derive_owner_secret(&base_key);
    let effective_key = auth::derive_owner_bound_key(
        &base_key, &owner_secret, key_version, issue_counter, crypto_timestamp,
    );

    let mut tag_expected = [0u8; auth::RNN_ENCRYPTED_TAG_SIZE];
    tag_expected.copy_from_slice(&data[payload_end..payload_end + auth::RNN_ENCRYPTED_TAG_SIZE]);

    let computed_tag = auth::compute_tag_v2(
        &effective_key, &nonce, key_version, issue_counter, crypto_timestamp,
        &data[payload_start..payload_end], plain_len,
    );
    if !constant_time_eq(&tag_expected, &computed_tag) {
        return enc_len;
    }

    let stream_key = auth::derive_stream_key_v2(&effective_key, &nonce);
    auth::xor_with_keystream_v2_in_place(
        &stream_key, &nonce, &mut data[payload_start..payload_end],
    );

    let mut stored_digest = [0u8; auth::RNN_ENCRYPTED_DIGEST_SIZE];
    stored_digest.copy_from_slice(
        &data[auth::RNN_ENCRYPTED_META_END..auth::RNN_ENCRYPTED_LEN_PREFIX],
    );
    if !verify_sha512(&data[payload_start..payload_end], &stored_digest) {
        return enc_len;
    }

    data.copy_within(payload_start..payload_end, 0);
    plain_len
}
