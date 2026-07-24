use crate::security::crypto::{constant_time_eq, sha256_bytes};

pub const ASYM_SECRET_KEY_LEN: usize = 32;
const ASYM_LAMPORT_BITS: usize = 32;
const ASYM_CHUNK_LEN: usize = 32;
pub const ASYM_PUBLIC_KEY_LEN: usize = ASYM_LAMPORT_BITS * 2 * ASYM_CHUNK_LEN;
pub const ASYM_SIGNATURE_LEN: usize = ASYM_LAMPORT_BITS * ASYM_CHUNK_LEN;
pub const RNN_ENCRYPTED_MAGIC_SIZE: usize = 4;
pub const RNN_ENCRYPTED_NONCE_SIZE: usize = 24;
pub const RNN_ENCRYPTED_KEY_VERSION_SIZE: usize = 4;
pub const RNN_ENCRYPTED_ISSUE_COUNTER_SIZE: usize = 8;
pub const RNN_ENCRYPTED_TIMESTAMP_SIZE: usize = 8;
pub const RNN_ENCRYPTED_PUBLIC_META_SIZE: usize = 32;
pub const RNN_ENCRYPTED_DIGEST_SIZE: usize = 64;
pub const RNN_ENCRYPTED_META_END: usize = RNN_ENCRYPTED_MAGIC_SIZE
    + RNN_ENCRYPTED_NONCE_SIZE
    + RNN_ENCRYPTED_KEY_VERSION_SIZE
    + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE
    + RNN_ENCRYPTED_TIMESTAMP_SIZE
    + RNN_ENCRYPTED_PUBLIC_META_SIZE;
pub const RNN_ENCRYPTED_LEN_PREFIX: usize = RNN_ENCRYPTED_META_END + RNN_ENCRYPTED_DIGEST_SIZE;
pub const RNN_ENCRYPTED_TAG_SIZE: usize = 32;
pub const RNN_ENCRYPTED_HEADER_SIZE: usize = RNN_ENCRYPTED_LEN_PREFIX + RNN_ENCRYPTED_TAG_SIZE;
pub const RNN_MIN_ACCEPTED_KEY_VERSION: u32 = 1;

pub(crate) const RNN_ENCRYPTED_MAGIC: [u8; RNN_ENCRYPTED_MAGIC_SIZE] = *b"RNN0";
const PUBLIC_META_VERSION: u8 = 1;
const PUBLIC_META_FLAG_HAS_BENCHMARK: u8 = 1 << 0;
const PUBLIC_META_FLAG_HAS_HEADER: u8 = 1 << 1;
const PUBLIC_META_FLAG_HAS_MODEL_NAME: u8 = 1 << 2;
const PUBLIC_META_FLAG_HAS_MODEL_PRECISION: u8 = 1 << 3;

const TLV_BLOB_TABLE: u8 = 0x03;
const MODEL_NAME_BLOB_DATA: &str = "model.name";
const MODEL_PRECISION_BLOB_DATA: &str = "model.precision";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RnnEncryptionMetadata {
    pub key_version: u32,
    pub issue_counter: u64,
    pub crypto_timestamp: u64,
    pub nonce: [u8; RNN_ENCRYPTED_NONCE_SIZE],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignatureError {
    InvalidPublicKey,
    InvalidSignature,
}

pub fn create_encryption_metadata(key_version: u32, issue_counter: u64, crypto_timestamp: u64, nonce: [u8; RNN_ENCRYPTED_NONCE_SIZE]) -> RnnEncryptionMetadata {
    RnnEncryptionMetadata {
        key_version,
        issue_counter,
        crypto_timestamp,
        nonce,
    }
}

pub fn metadata_for_payload(key_version: u32, issue_counter: u64, crypto_timestamp: u64, nonce: [u8; RNN_ENCRYPTED_NONCE_SIZE]) -> RnnEncryptionMetadata {
    create_encryption_metadata(key_version, issue_counter, crypto_timestamp, nonce)
}

pub fn create_signature_error_invalid_public_key() -> SignatureError {
    SignatureError::InvalidPublicKey
}

pub fn create_signature_error_invalid_signature() -> SignatureError {
    SignatureError::InvalidSignature
}

pub fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    const BLOCK: usize = 64;
    let mut k0 = [0u8; BLOCK];

    if key.len() > BLOCK {
        let mut key_hash = [0u8; 32];
        sha256_bytes(key, &mut key_hash);
        k0[..32].copy_from_slice(&key_hash);
    } else {
        k0[..key.len()].copy_from_slice(key);
    }

    let mut ipad = [0u8; BLOCK];
    let mut opad = [0u8; BLOCK];
    for i in 0..BLOCK {
        ipad[i] = k0[i] ^ 0x36;
        opad[i] = k0[i] ^ 0x5c;
    }

    let mut inner_buf = [0u8; BLOCK + 32];
    inner_buf[..BLOCK].copy_from_slice(&ipad);
    let mut inner = [0u8; 32];
    if data.len() <= 32 {
        inner_buf[BLOCK..BLOCK + data.len()].copy_from_slice(data);
        sha256_bytes(&inner_buf[..BLOCK + data.len()], &mut inner);
    } else {
        let mut mixed = [0u8; 32];
        sha256_bytes(data, &mut mixed);
        inner_buf[BLOCK..].copy_from_slice(&mixed);
        sha256_bytes(&inner_buf, &mut inner);
    }

    let mut outer_buf = [0u8; BLOCK + 32];
    outer_buf[..BLOCK].copy_from_slice(&opad);
    outer_buf[BLOCK..].copy_from_slice(&inner);
    let mut out = [0u8; 32];
    sha256_bytes(&outer_buf, &mut out);
    out
}

pub fn verify_hmac_sha256(key: &[u8], data: &[u8], expected: &[u8; 32]) -> bool {
    let computed = hmac_sha256(key, data);
    constant_time_eq(&computed, expected)
}

pub fn compute_hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    hmac_sha256(key, data)
}

pub fn lamport_private_elem(
    secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    bit_index: usize,
    bit_value: u8,
) -> [u8; 32] {
    let mut seed = [0u8; ASYM_SECRET_KEY_LEN + 4 + 1];
    seed[..ASYM_SECRET_KEY_LEN].copy_from_slice(secret_key);
    seed[ASYM_SECRET_KEY_LEN..ASYM_SECRET_KEY_LEN + 4]
        .copy_from_slice(&(bit_index as u32).to_le_bytes());
    seed[ASYM_SECRET_KEY_LEN + 4] = bit_value;

    let mut out = [0u8; 32];
    sha256_bytes(&seed, &mut out);
    out
}

pub fn ed25519_public_from_secret(
    secret_key: &[u8; ASYM_SECRET_KEY_LEN],
) -> [u8; ASYM_PUBLIC_KEY_LEN] {
    let mut public_key = [0u8; ASYM_PUBLIC_KEY_LEN];
    for bit_index in 0..ASYM_LAMPORT_BITS {
        for bit_value in 0..=1u8 {
            let private_elem = lamport_private_elem(secret_key, bit_index, bit_value);
            let mut public_elem = [0u8; ASYM_CHUNK_LEN];
            sha256_bytes(&private_elem, &mut public_elem);
            let offset = (bit_index * 2 + bit_value as usize) * ASYM_CHUNK_LEN;
            public_key[offset..offset + ASYM_CHUNK_LEN].copy_from_slice(&public_elem);
        }
    }
    public_key
}

pub fn ed25519_sign(
    secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    message: &[u8],
) -> [u8; ASYM_SIGNATURE_LEN] {
    let mut digest = [0u8; 32];
    sha256_bytes(message, &mut digest);

    let mut signature = [0u8; ASYM_SIGNATURE_LEN];
    for bit_index in 0..ASYM_LAMPORT_BITS {
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let bit_value = (digest[byte_index] >> shift) & 1;
        let private_elem = lamport_private_elem(secret_key, bit_index, bit_value);
        let offset = bit_index * ASYM_CHUNK_LEN;
        signature[offset..offset + ASYM_CHUNK_LEN].copy_from_slice(&private_elem);
    }

    signature
}

pub fn sign_message(secret_key: &[u8; ASYM_SECRET_KEY_LEN], message: &[u8]) -> [u8; ASYM_SIGNATURE_LEN] {
    ed25519_sign(secret_key, message)
}

pub fn ed25519_verify(
    public_key: &[u8; ASYM_PUBLIC_KEY_LEN],
    message: &[u8],
    signature: &[u8; ASYM_SIGNATURE_LEN],
) -> Result<(), SignatureError> {
    if public_key.iter().all(|&byte| byte == 0) {
        return Err(create_signature_error_invalid_public_key());
    }
    let mut digest = [0u8; 32];
    sha256_bytes(message, &mut digest);

    for bit_index in 0..ASYM_LAMPORT_BITS {
        let byte_index = bit_index / 8;
        let shift = bit_index % 8;
        let bit_value = ((digest[byte_index] >> shift) & 1) as usize;

        let sig_offset = bit_index * ASYM_CHUNK_LEN;
        let pub_offset = (bit_index * 2 + bit_value) * ASYM_CHUNK_LEN;

        let mut sig_hash = [0u8; ASYM_CHUNK_LEN];
        sha256_bytes(
            &signature[sig_offset..sig_offset + ASYM_CHUNK_LEN],
            &mut sig_hash,
        );

        let expected = &public_key[pub_offset..pub_offset + ASYM_CHUNK_LEN];
        if !constant_time_eq(&sig_hash, expected) {
            return Err(create_signature_error_invalid_signature());
        }
    }

    Ok(())
}

pub fn verify_signature(public_key: &[u8; ASYM_PUBLIC_KEY_LEN], message: &[u8], signature: &[u8; ASYM_SIGNATURE_LEN]) -> bool {
    ed25519_verify(public_key, message, signature).is_ok()
}

pub fn is_encrypted_rnn(bytes: &[u8]) -> bool {
    bytes.len() >= RNN_ENCRYPTED_HEADER_SIZE
        && constant_time_eq(&bytes[..RNN_ENCRYPTED_MAGIC_SIZE], &RNN_ENCRYPTED_MAGIC)
}

pub fn encrypted_rnn_size(plain_len: usize) -> Option<usize> {
    plain_len.checked_add(RNN_ENCRYPTED_HEADER_SIZE)
}

pub fn extract_rnn_crypto_timestamp(encrypted: &[u8]) -> Option<u64> {
    if encrypted.len() < RNN_ENCRYPTED_HEADER_SIZE {
        return None;
    }
    if !constant_time_eq(&encrypted[..RNN_ENCRYPTED_MAGIC_SIZE], &RNN_ENCRYPTED_MAGIC) {
        return None;
    }
    let ts_start = RNN_ENCRYPTED_MAGIC_SIZE
        + RNN_ENCRYPTED_NONCE_SIZE
        + RNN_ENCRYPTED_KEY_VERSION_SIZE
        + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
    let ts_end = ts_start + RNN_ENCRYPTED_TIMESTAMP_SIZE;
    let ts: [u8; RNN_ENCRYPTED_TIMESTAMP_SIZE] = encrypted[ts_start..ts_end].try_into().ok()?;
    Some(u64::from_le_bytes(ts))
}

pub fn extract_rnn_key_version(encrypted: &[u8]) -> Option<u32> {
    if encrypted.len() < RNN_ENCRYPTED_HEADER_SIZE {
        return None;
    }
    if !constant_time_eq(&encrypted[..RNN_ENCRYPTED_MAGIC_SIZE], &RNN_ENCRYPTED_MAGIC) {
        return None;
    }
    let ver_start = RNN_ENCRYPTED_MAGIC_SIZE + RNN_ENCRYPTED_NONCE_SIZE;
    let ver_end = ver_start + RNN_ENCRYPTED_KEY_VERSION_SIZE;
    let ver: [u8; RNN_ENCRYPTED_KEY_VERSION_SIZE] =
        encrypted[ver_start..ver_end].try_into().ok()?;
    Some(u32::from_le_bytes(ver))
}

pub fn extract_rnn_issue_counter(encrypted: &[u8]) -> Option<u64> {
    if encrypted.len() < RNN_ENCRYPTED_HEADER_SIZE {
        return None;
    }
    if !constant_time_eq(&encrypted[..RNN_ENCRYPTED_MAGIC_SIZE], &RNN_ENCRYPTED_MAGIC) {
        return None;
    }
    let ctr_start =
        RNN_ENCRYPTED_MAGIC_SIZE + RNN_ENCRYPTED_NONCE_SIZE + RNN_ENCRYPTED_KEY_VERSION_SIZE;
    let ctr_end = ctr_start + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
    let ctr: [u8; RNN_ENCRYPTED_ISSUE_COUNTER_SIZE] =
        encrypted[ctr_start..ctr_end].try_into().ok()?;
    Some(u64::from_le_bytes(ctr))
}

pub fn extract_rnn_public_has_benchmark(encrypted: &[u8]) -> Option<bool> {
    let meta = extract_rnn_public_meta(encrypted)?;
    if meta[0] != PUBLIC_META_VERSION {
        return None;
    }
    Some((meta[1] & PUBLIC_META_FLAG_HAS_BENCHMARK) != 0)
}

pub fn extract_rnn_public_has_model_name(encrypted: &[u8]) -> Option<bool> {
    let meta = extract_rnn_public_meta(encrypted)?;
    if meta[0] != PUBLIC_META_VERSION {
        return None;
    }
    Some((meta[1] & PUBLIC_META_FLAG_HAS_MODEL_NAME) != 0)
}

pub fn extract_rnn_public_has_model_precision(encrypted: &[u8]) -> Option<bool> {
    let meta = extract_rnn_public_meta(encrypted)?;
    if meta[0] != PUBLIC_META_VERSION {
        return None;
    }
    Some((meta[1] & PUBLIC_META_FLAG_HAS_MODEL_PRECISION) != 0)
}

pub fn extract_rnn_public_header_summary(
    encrypted: &[u8],
) -> Option<(u8, u32, u32, u32, u32, u32, u32)> {
    let meta = extract_rnn_public_meta(encrypted)?;
    if meta[0] != PUBLIC_META_VERSION || (meta[1] & PUBLIC_META_FLAG_HAS_HEADER) == 0 {
        return None;
    }

    let dtype = meta[2];
    let layer_count = u32::from_le_bytes([meta[4], meta[5], meta[6], meta[7]]);
    let total_neurons = u32::from_le_bytes([meta[8], meta[9], meta[10], meta[11]]);
    let weights_len = u32::from_le_bytes([meta[12], meta[13], meta[14], meta[15]]);
    let biases_len = u32::from_le_bytes([meta[16], meta[17], meta[18], meta[19]]);
    let blob_count = u32::from_le_bytes([meta[20], meta[21], meta[22], meta[23]]);
    let flags = u32::from_le_bytes([meta[24], meta[25], meta[26], meta[27]]);

    Some((
        dtype,
        layer_count,
        total_neurons,
        weights_len,
        biases_len,
        blob_count,
        flags,
    ))
}

fn extract_rnn_public_meta(encrypted: &[u8]) -> Option<&[u8]> {
    if encrypted.len() < RNN_ENCRYPTED_HEADER_SIZE {
        return None;
    }
    if !constant_time_eq(&encrypted[..RNN_ENCRYPTED_MAGIC_SIZE], &RNN_ENCRYPTED_MAGIC) {
        return None;
    }
    let meta_start = RNN_ENCRYPTED_MAGIC_SIZE
        + RNN_ENCRYPTED_NONCE_SIZE
        + RNN_ENCRYPTED_KEY_VERSION_SIZE
        + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE
        + RNN_ENCRYPTED_TIMESTAMP_SIZE;
    let meta_end = meta_start + RNN_ENCRYPTED_PUBLIC_META_SIZE;
    encrypted.get(meta_start..meta_end)
}

pub fn encrypt_rnn_payload(key: &[u8], plaintext: &[u8], out: &mut [u8]) -> Option<usize> {
    encrypt_rnn_payload_owner_bound(key, &[0u8; ASYM_SECRET_KEY_LEN], 1, 1, 1, plaintext, out)
}

pub fn encrypt_payload(key: &[u8], plaintext: &[u8], out: &mut [u8]) -> Option<usize> {
    encrypt_rnn_payload(key, plaintext, out)
}

pub fn encrypt_rnn_payload_owner_bound(
    key: &[u8],
    owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    key_version: u32,
    issue_counter: u64,
    crypto_timestamp: u64,
    plaintext: &[u8],
    out: &mut [u8],
) -> Option<usize> {
    let nonce = derive_nonce_v2(
        key,
        owner_secret_key,
        key_version,
        issue_counter,
        crypto_timestamp,
        plaintext,
    );
    let metadata = metadata_for_payload(key_version, issue_counter, crypto_timestamp, nonce);
    encrypt_rnn_payload_owner_bound_with_nonce(key, owner_secret_key, metadata, plaintext, out)
}

pub fn encrypt_rnn_payload_owner_bound_with_nonce(
    key: &[u8],
    owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    metadata: RnnEncryptionMetadata,
    plaintext: &[u8],
    out: &mut [u8],
) -> Option<usize> {
    if key.is_empty() {
        return None;
    }
    if plaintext.len() > u64::MAX as usize {
        return None;
    }

    let total = encrypted_rnn_size(plaintext.len())?;
    if out.len() < total {
        return None;
    }

    out[..RNN_ENCRYPTED_MAGIC_SIZE].copy_from_slice(&RNN_ENCRYPTED_MAGIC);
    if metadata.key_version == 0 || metadata.issue_counter == 0 || metadata.crypto_timestamp == 0 {
        return None;
    }
    let nonce_start = RNN_ENCRYPTED_MAGIC_SIZE;
    let nonce_end = nonce_start + RNN_ENCRYPTED_NONCE_SIZE;
    let ver_start = nonce_end;
    let ver_end = ver_start + RNN_ENCRYPTED_KEY_VERSION_SIZE;
    let ctr_start = ver_end;
    let ctr_end = ctr_start + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
    let ts_start = ctr_end;
    let ts_end = ts_start + RNN_ENCRYPTED_TIMESTAMP_SIZE;
    out[nonce_start..nonce_end].copy_from_slice(&metadata.nonce);
    out[ver_start..ver_end].copy_from_slice(&metadata.key_version.to_le_bytes());
    out[ctr_start..ctr_end].copy_from_slice(&metadata.issue_counter.to_le_bytes());
    out[ts_start..ts_end].copy_from_slice(&metadata.crypto_timestamp.to_le_bytes());

    let public_meta = build_public_meta_from_plaintext(plaintext);
    out[ts_end..RNN_ENCRYPTED_LEN_PREFIX].copy_from_slice(&public_meta);

    let digest = compute_hmac_sha256(key, &out[..RNN_ENCRYPTED_LEN_PREFIX]);
    out[RNN_ENCRYPTED_LEN_PREFIX..RNN_ENCRYPTED_LEN_PREFIX + 32].copy_from_slice(&digest);

    let owner_sig = sign_message(owner_secret_key, &out[..RNN_ENCRYPTED_LEN_PREFIX]);
    out[RNN_ENCRYPTED_LEN_PREFIX + 32..RNN_ENCRYPTED_HEADER_SIZE].copy_from_slice(&owner_sig);

    out[RNN_ENCRYPTED_HEADER_SIZE..RNN_ENCRYPTED_HEADER_SIZE + plaintext.len()].copy_from_slice(plaintext);

    Some(total)
}

pub fn encrypt_payload_owner_bound(key: &[u8], owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN], key_version: u32, issue_counter: u64, crypto_timestamp: u64, plaintext: &[u8], out: &mut [u8]) -> Option<usize> {
    encrypt_rnn_payload_owner_bound(key, owner_secret_key, key_version, issue_counter, crypto_timestamp, plaintext, out)
}

pub(crate) fn build_public_meta_from_plaintext(plaintext: &[u8]) -> [u8; RNN_ENCRYPTED_PUBLIC_META_SIZE] {
    let mut out = [0u8; RNN_ENCRYPTED_PUBLIC_META_SIZE];
    out[0] = PUBLIC_META_VERSION;

    if plaintext.len() < 12 || !constant_time_eq(&plaintext[0..4], b"RNN\x00") {
        return out;
    }

    let header_size =
        u32::from_le_bytes([plaintext[8], plaintext[9], plaintext[10], plaintext[11]]) as usize;
    if header_size < 12 || header_size > plaintext.len() {
        return out;
    }

    let mut cursor = 12usize;
    while cursor + 5 <= header_size {
        let tlv_type = plaintext[cursor];
        cursor += 1;
        let tlv_len = u32::from_le_bytes([
            plaintext[cursor],
            plaintext[cursor + 1],
            plaintext[cursor + 2],
            plaintext[cursor + 3],
        ]) as usize;
        cursor += 4;
        let end = match cursor.checked_add(tlv_len) {
            Some(v) if v <= header_size => v,
            _ => return out,
        };

        if tlv_type == 0x04 {
            out[1] |= PUBLIC_META_FLAG_HAS_BENCHMARK;
        }
        if tlv_type == 0x05 {
            if let Some((
                dtype,
                layer_count,
                total_neurons,
                weights_len,
                biases_len,
                blob_count,
                flags,
            )) = decode_network_summary_payload(&plaintext[cursor..end])
            {
                out[1] |= PUBLIC_META_FLAG_HAS_HEADER;
                out[2] = dtype;
                out[4..8].copy_from_slice(&layer_count.to_le_bytes());
                out[8..12].copy_from_slice(&total_neurons.to_le_bytes());
                out[12..16].copy_from_slice(&weights_len.to_le_bytes());
                out[16..20].copy_from_slice(&biases_len.to_le_bytes());
                out[20..24].copy_from_slice(&blob_count.to_le_bytes());
                out[24..28].copy_from_slice(&flags.to_le_bytes());
            }
        }
        if tlv_type == TLV_BLOB_TABLE {
            let (has_model_name, has_model_precision) =
                parse_blob_table_presence(&plaintext[cursor..end]);
            if has_model_name {
                out[1] |= PUBLIC_META_FLAG_HAS_MODEL_NAME;
            }
            if has_model_precision {
                out[1] |= PUBLIC_META_FLAG_HAS_MODEL_PRECISION;
            }
        }
        cursor = end;
    }

    out
}

fn parse_blob_table_presence(table_payload: &[u8]) -> (bool, bool) {
    let mut has_model_name = false;
    let mut has_model_precision = false;

    let mut cursor = 0usize;
    while cursor + 2 <= table_payload.len() {
        let name_len =
            u16::from_le_bytes([table_payload[cursor], table_payload[cursor + 1]]) as usize;
        cursor += 2;
        let name_end = match cursor.checked_add(name_len) {
            Some(v) if v <= table_payload.len() => v,
            _ => return (has_model_name, has_model_precision),
        };
        let name = &table_payload[cursor..name_end];
        cursor = name_end;

        if cursor + 2 > table_payload.len() {
            return (has_model_name, has_model_precision);
        }
        let dtype = table_payload[cursor];
        let ndim = table_payload[cursor + 1] as usize;
        cursor += 2;
        if !matches!(dtype, 0 | 1) {
            return (has_model_name, has_model_precision);
        }

        let dims_bytes = match ndim.checked_mul(4) {
            Some(v) => v,
            None => return (has_model_name, has_model_precision),
        };
        let after_dims = match cursor.checked_add(dims_bytes) {
            Some(v) if v <= table_payload.len() => v,
            _ => return (has_model_name, has_model_precision),
        };
        cursor = after_dims;

        let after_meta = match cursor.checked_add(8 + 8 + 32) {
            Some(v) if v <= table_payload.len() => v,
            _ => return (has_model_name, has_model_precision),
        };
        cursor = after_meta;

        if name == MODEL_NAME_BLOB_DATA.as_bytes() {
            has_model_name = true;
        }
        if name == MODEL_PRECISION_BLOB_DATA.as_bytes() {
            has_model_precision = true;
        }
    }

    (has_model_name, has_model_precision)
}

fn decode_network_summary_payload(payload: &[u8]) -> Option<(u8, u32, u32, u32, u32, u32, u32)> {
    if payload.len() < 2 {
        return None;
    }
    if payload[0] != 1 || payload[1] != 1 {
        return None;
    }

    let compressed = &payload[2..];
    if !compressed.len().is_multiple_of(2) {
        return None;
    }

    let mut plain = [0u8; 30];
    let mut plain_len = 0usize;
    let mut idx = 0usize;
    while idx < compressed.len() {
        let run = compressed[idx] as usize;
        let value = compressed[idx + 1];
        if run == 0 {
            return None;
        }
        plain_len = plain_len.checked_add(run)?;
        if plain_len <= 30 {
            let start = plain_len - run;
            let end = plain_len;
            plain[start..end].fill(value);
        } else if plain_len - run < 30 {
            let start = plain_len - run;
            let end = 30usize;
            plain[start..end].fill(value);
        }
        idx += 2;
    }

    if plain_len < 30 || !constant_time_eq(&plain[0..4], b"S5D0") || plain[4] != 1 {
        return None;
    }

    let dtype = plain[5];
    let layer_count = u32::from_le_bytes([plain[6], plain[7], plain[8], plain[9]]);
    let total_neurons = u32::from_le_bytes([plain[10], plain[11], plain[12], plain[13]]);
    let weights_len = u32::from_le_bytes([plain[14], plain[15], plain[16], plain[17]]);
    let biases_len = u32::from_le_bytes([plain[18], plain[19], plain[20], plain[21]]);
    let blob_count = u32::from_le_bytes([plain[22], plain[23], plain[24], plain[25]]);
    let flags = u32::from_le_bytes([plain[26], plain[27], plain[28], plain[29]]);
    Some((
        dtype,
        layer_count,
        total_neurons,
        weights_len,
        biases_len,
        blob_count,
        flags,
    ))
}

pub fn decrypt_rnn_payload(key: &[u8], encrypted: &[u8], out: &mut [u8]) -> Option<usize> {
    decrypt_rnn_payload_owner_bound(key, &[0u8; ASYM_SECRET_KEY_LEN], encrypted, out)
}

pub fn decrypt_payload(key: &[u8], encrypted: &[u8], out: &mut [u8]) -> Option<usize> {
    decrypt_rnn_payload(key, encrypted, out)
}

pub fn decrypt_rnn_payload_owner_bound(
    key: &[u8],
    owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    encrypted: &[u8],
    out: &mut [u8],
) -> Option<usize> {
    if key.is_empty() {
        return None;
    }

    if encrypted.len() >= RNN_ENCRYPTED_HEADER_SIZE
        && constant_time_eq(&encrypted[..RNN_ENCRYPTED_MAGIC_SIZE], &RNN_ENCRYPTED_MAGIC)
    {
        let plain_len = encrypted.len().checked_sub(RNN_ENCRYPTED_HEADER_SIZE)?;
        if out.len() < plain_len {
            return None;
        }

        let nonce_start = RNN_ENCRYPTED_MAGIC_SIZE;
        let nonce_end = nonce_start + RNN_ENCRYPTED_NONCE_SIZE;
        let ver_start = nonce_end;
        let ver_end = ver_start + RNN_ENCRYPTED_KEY_VERSION_SIZE;
        let ctr_start = ver_end;
        let ctr_end = ctr_start + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
        let ts_start = ctr_end;
        let ts_end = ts_start + RNN_ENCRYPTED_TIMESTAMP_SIZE;
        let public_meta_start = ts_end;
        let public_meta_end = public_meta_start + RNN_ENCRYPTED_PUBLIC_META_SIZE;
        let digest_end = public_meta_end + RNN_ENCRYPTED_DIGEST_SIZE;
        if encrypted.get(public_meta_start..public_meta_end).is_none()
            || encrypted.get(public_meta_end..digest_end).is_none()
        {
            return None;
        }

        let nonce: [u8; RNN_ENCRYPTED_NONCE_SIZE] = encrypted[nonce_start..nonce_end].try_into().ok()?;
        let key_version = u32::from_le_bytes([
            encrypted[ver_start],
            encrypted[ver_start + 1],
            encrypted[ver_start + 2],
            encrypted[ver_start + 3],
        ]);
        let issue_counter = u64::from_le_bytes([
            encrypted[ctr_start],
            encrypted[ctr_start + 1],
            encrypted[ctr_start + 2],
            encrypted[ctr_start + 3],
            encrypted[ctr_start + 4],
            encrypted[ctr_start + 5],
            encrypted[ctr_start + 6],
            encrypted[ctr_start + 7],
        ]);
        let crypto_timestamp = u64::from_le_bytes([
            encrypted[ts_start],
            encrypted[ts_start + 1],
            encrypted[ts_start + 2],
            encrypted[ts_start + 3],
            encrypted[ts_start + 4],
            encrypted[ts_start + 5],
            encrypted[ts_start + 6],
            encrypted[ts_start + 7],
        ]);

        let effective_key = derive_owner_bound_key(
            key,
            owner_secret_key,
            key_version,
            issue_counter,
            crypto_timestamp,
        );

        let metadata = create_encryption_metadata(
            key_version,
            issue_counter,
            crypto_timestamp,
            nonce,
        );
        let metadata_is_supported = metadata.key_version >= RNN_MIN_ACCEPTED_KEY_VERSION
            && metadata.issue_counter != 0
            && metadata.crypto_timestamp != 0
            && metadata.nonce.iter().any(|&byte| byte != 0);
        if !metadata_is_supported {
            return None;
        }

        let header_prefix = &encrypted[..RNN_ENCRYPTED_LEN_PREFIX];
        let stored_digest = encrypted[RNN_ENCRYPTED_LEN_PREFIX..RNN_ENCRYPTED_LEN_PREFIX + RNN_ENCRYPTED_DIGEST_SIZE]
            .try_into()
            .ok()?;
        if !verify_hmac_sha256(key, header_prefix, &stored_digest) {
            return None;
        }

        let owner_public_key = ed25519_public_from_secret(owner_secret_key);
        let sig_start = RNN_ENCRYPTED_LEN_PREFIX + RNN_ENCRYPTED_DIGEST_SIZE;
        let sig_end = sig_start + ASYM_SIGNATURE_LEN;
        let signature: [u8; ASYM_SIGNATURE_LEN] = encrypted[sig_start..sig_end].try_into().ok()?;
        if verify_signature(&owner_public_key, header_prefix, &signature) {
            let payload_start = RNN_ENCRYPTED_HEADER_SIZE;
            let payload_end = payload_start + plain_len;
            out[..plain_len].copy_from_slice(&encrypted[payload_start..payload_end]);

            let stream_key = derive_stream_key_v2(&effective_key, &nonce);
            xor_with_keystream_v2_in_place(&stream_key, &nonce, &mut out[..plain_len]);

            let tag_start = RNN_ENCRYPTED_LEN_PREFIX;
            let tag_end = tag_start + RNN_ENCRYPTED_TAG_SIZE;
            let expected_tag = &encrypted[tag_start..tag_end];
            let computed_tag = compute_tag_v2(
                &effective_key,
                &nonce,
                key_version,
                issue_counter,
                crypto_timestamp,
                &out[..plain_len],
                plain_len,
            );

            if !constant_time_eq(expected_tag, &computed_tag) {
                return None;
            }

            Some(plain_len)
        } else {
            None
        }
    } else {
        None
    }
}

pub fn decrypt_payload_owner_bound(key: &[u8], owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN], encrypted: &[u8], out: &mut [u8]) -> Option<usize> {
    decrypt_rnn_payload_owner_bound(key, owner_secret_key, encrypted, out)
}

pub(crate) fn derive_owner_bound_key(
    key: &[u8],
    owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    key_version: u32,
    issue_counter: u64,
    crypto_timestamp: u64,
) -> [u8; 32] {
    let mut bind_input = [0u8; ASYM_SECRET_KEY_LEN + 4 + 8 + 8];
    bind_input[..ASYM_SECRET_KEY_LEN].copy_from_slice(owner_secret_key);
    bind_input[ASYM_SECRET_KEY_LEN..ASYM_SECRET_KEY_LEN + 4]
        .copy_from_slice(&key_version.to_le_bytes());
    bind_input[ASYM_SECRET_KEY_LEN + 4..ASYM_SECRET_KEY_LEN + 12]
        .copy_from_slice(&issue_counter.to_le_bytes());
    bind_input[ASYM_SECRET_KEY_LEN + 12..].copy_from_slice(&crypto_timestamp.to_le_bytes());
    let bind_ctx = hmac_sha256(key, b"rnn-owner-bind-v1");
    hmac_sha256(&bind_ctx, &bind_input)
}

pub(crate) fn derive_nonce_v2(
    key: &[u8],
    owner_secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    key_version: u32,
    issue_counter: u64,
    crypto_timestamp: u64,
    plaintext: &[u8],
) -> [u8; RNN_ENCRYPTED_NONCE_SIZE] {
    let mut plain_hash = [0u8; 32];
    sha256_bytes(plaintext, &mut plain_hash);

    let mut nonce_input = [0u8; 60];
    nonce_input[..32].copy_from_slice(&plain_hash);
    nonce_input[32..40].copy_from_slice(&(plaintext.len() as u64).to_le_bytes());
    nonce_input[40..44].copy_from_slice(&key_version.to_le_bytes());
    nonce_input[44..52].copy_from_slice(&issue_counter.to_le_bytes());
    nonce_input[52..60].copy_from_slice(&crypto_timestamp.to_le_bytes());

    let owner_bind = derive_owner_bound_key(
        key,
        owner_secret_key,
        key_version,
        issue_counter,
        crypto_timestamp,
    );
    let nonce_key = hmac_sha256(&owner_bind, b"rnn-nonce-v2");
    let nonce_full = hmac_sha256(&nonce_key, &nonce_input);
    let mut nonce = [0u8; RNN_ENCRYPTED_NONCE_SIZE];
    nonce.copy_from_slice(&nonce_full[..RNN_ENCRYPTED_NONCE_SIZE]);
    nonce
}

pub(crate) fn derive_stream_key_v2(
    owner_bound_key: &[u8; 32],
    nonce: &[u8; RNN_ENCRYPTED_NONCE_SIZE],
) -> [u8; 32] {
    let mut info = [0u8; 32];
    info[..RNN_ENCRYPTED_NONCE_SIZE].copy_from_slice(nonce);
    info[RNN_ENCRYPTED_NONCE_SIZE..].copy_from_slice(b"strm-v2!");
    hmac_sha256(owner_bound_key, &info)
}

pub(crate) fn compute_tag_v2(
    owner_bound_key: &[u8; 32],
    nonce: &[u8; RNN_ENCRYPTED_NONCE_SIZE],
    key_version: u32,
    issue_counter: u64,
    crypto_timestamp: u64,
    ciphertext: &[u8],
    plain_len: usize,
) -> [u8; RNN_ENCRYPTED_TAG_SIZE] {
    let mut cipher_hash = [0u8; 32];
    sha256_bytes(ciphertext, &mut cipher_hash);

    let mut tag_input = [0u8; 88];
    tag_input[..RNN_ENCRYPTED_MAGIC_SIZE].copy_from_slice(&RNN_ENCRYPTED_MAGIC);
    let nonce_start = RNN_ENCRYPTED_MAGIC_SIZE;
    let nonce_end = nonce_start + RNN_ENCRYPTED_NONCE_SIZE;
    let ver_start = nonce_end;
    let ver_end = ver_start + RNN_ENCRYPTED_KEY_VERSION_SIZE;
    let ctr_start = ver_end;
    let ctr_end = ctr_start + RNN_ENCRYPTED_ISSUE_COUNTER_SIZE;
    let ts_start = ctr_end;
    let ts_end = ts_start + RNN_ENCRYPTED_TIMESTAMP_SIZE;
    tag_input[nonce_start..nonce_end].copy_from_slice(nonce);
    tag_input[ver_start..ver_end].copy_from_slice(&key_version.to_le_bytes());
    tag_input[ctr_start..ctr_end].copy_from_slice(&issue_counter.to_le_bytes());
    tag_input[ts_start..ts_end].copy_from_slice(&crypto_timestamp.to_le_bytes());
    tag_input[ts_end..ts_end + 32].copy_from_slice(&cipher_hash);
    tag_input[ts_end + 32..].copy_from_slice(&(plain_len as u64).to_le_bytes());

    let tag_key = hmac_sha256(owner_bound_key, b"rnn-tag-v2");
    hmac_sha256(&tag_key, &tag_input)
}

pub(crate) fn xor_with_keystream_v2_in_place(
    stream_key: &[u8; 32],
    nonce: &[u8; RNN_ENCRYPTED_NONCE_SIZE],
    data: &mut [u8],
) {
    let mut ctr = 0u64;
    let mut offset = 0usize;
    while offset < data.len() {
        let block = opaque_keystream_block_v2(stream_key, nonce, ctr);
        let take = core::cmp::min(32, data.len() - offset);
        for i in 0..take {
            data[offset + i] ^= block[i];
        }
        ctr = ctr.wrapping_add(1);
        offset += take;
    }
}

fn opaque_keystream_block_v2(
    stream_key: &[u8; 32],
    nonce: &[u8; RNN_ENCRYPTED_NONCE_SIZE],
    counter: u64,
) -> [u8; 32] {
    let mut seed = [0u8; 64];
    seed[0..32].copy_from_slice(stream_key);
    seed[32..32 + RNN_ENCRYPTED_NONCE_SIZE].copy_from_slice(nonce);
    seed[32 + RNN_ENCRYPTED_NONCE_SIZE..64].copy_from_slice(&counter.to_le_bytes());
    let mut out = [0u8; 32];
    sha256_bytes(&seed, &mut out);
    out
}
