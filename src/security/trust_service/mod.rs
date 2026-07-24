use crate::security::crypto::ASYM_PUBLIC_KEY_LEN;
use crate::security::crypto::{
    create_encryption_metadata, ed25519_public_from_secret, ed25519_sign, verify_signature,
    ASYM_SECRET_KEY_LEN, ASYM_SIGNATURE_LEN,
};
use crate::observability::benchmark::{encode_benchmark_blob, BenchmarkMetrics};
use crate::format::model_config::ingest::{
    AUTH_DISTRIBUTION_POLICY_BLOB_DATA, AUTH_ED25519_PUBKEY_BLOB_DATA, AUTH_ED25519_SIG_BLOB_DATA,
    AUTH_HMAC_SHA256_BLOB_DATA, BLOB_BIASES, BLOB_LAYER_META, BLOB_WEIGHTS,
    DISTRIBUTION_POLICY_PUBLISHER_SHARED, DISTRIBUTION_POLICY_USER_DEVICE_LOCKED,
    TLV_HEADER_BENCHMARK, TLV_HEADER_MODEL_NAME,
};
use crate::format::model_format::container::assemble_rnn_container;
use crate::format::model_format::{header_tlv_payload, BlobDesc};
use crate::format::rnn_format::parser::parse_rnn_from_bytes;
use crate::base::scratch::Scratch;

pub const ACTIVATION_TOKEN_MAX_LEN: usize = 256;

const SIGN_MAX_BLOBS: usize = 32;
const SIGN_PARSE_SCRATCH_LEN: usize = 16384;
const SIGN_MESSAGE_CAP: usize = 262144;
const SIGN_BENCHMARK_CAP: usize = 256;

struct CanonicalMessage {
    data: [u8; SIGN_MESSAGE_CAP],
    len: usize,
}

impl CanonicalMessage {
    fn new() -> Self {
        Self { data: [0u8; SIGN_MESSAGE_CAP], len: 0 }
    }

    fn extend(&mut self, bytes: &[u8]) -> Result<(), TrustServiceError> {
        let next = self
            .len
            .checked_add(bytes.len())
            .ok_or(TrustServiceError::InvalidChallenge)?;
        if next > SIGN_MESSAGE_CAP {
            return Err(TrustServiceError::InvalidChallenge);
        }
        self.data[self.len..next].copy_from_slice(bytes);
        self.len = next;
        Ok(())
    }

    fn push(&mut self, byte: u8) -> Result<(), TrustServiceError> {
        if self.len >= SIGN_MESSAGE_CAP {
            return Err(TrustServiceError::InvalidChallenge);
        }
        self.data[self.len] = byte;
        self.len += 1;
        Ok(())
    }

    fn as_slice(&self) -> &[u8] {
        &self.data[..self.len]
    }
}

pub fn sign_model_distribution(
    container: &[u8],
    secret_key: &[u8; ASYM_SECRET_KEY_LEN],
    device_id: &[u8],
    distribution_policy: u8,
    out: &mut [u8],
) -> Result<usize, TrustServiceError> {
    let bind_device: &[u8] = match distribution_policy {
        DISTRIBUTION_POLICY_PUBLISHER_SHARED => &[],
        DISTRIBUTION_POLICY_USER_DEVICE_LOCKED => {
            if device_id.is_empty() {
                return Err(TrustServiceError::InvalidChallenge);
            }
            device_id
        }
        _ => return Err(TrustServiceError::InvalidChallenge),
    };
    if bind_device.len() > u16::MAX as usize {
        return Err(TrustServiceError::InvalidChallenge);
    }

    let mut parse_scratch = [0u8; SIGN_PARSE_SCRATCH_LEN];
    let mut scratch = Scratch::new(&mut parse_scratch);
    let handle = parse_rnn_from_bytes(container, &mut scratch)
        .map_err(|_| TrustServiceError::InvalidChallenge)?;

    let blob_count = handle.blobs.len();
    if blob_count > SIGN_MAX_BLOBS - 3 {
        return Err(TrustServiceError::InvalidChallenge);
    }

    let mut records: [BlobDesc; SIGN_MAX_BLOBS] = [BlobDesc {
        name: "",
        dtype: 0,
        dims: [0u32, 0u32],
        ndim: 1,
        payload: &[],
    }; SIGN_MAX_BLOBS];
    let mut records_count = 0usize;

    for i in 0..blob_count {
        let name = handle
            .blob_name(i)
            .ok_or(TrustServiceError::InvalidChallenge)?;
        let meta = handle
            .blobs
            .get(i)
            .ok_or(TrustServiceError::InvalidChallenge)?;
        if meta.ndim == 0 || meta.ndim > 2 {
            return Err(TrustServiceError::InvalidChallenge);
        }
        let start = usize::try_from(meta.offset).map_err(|_| TrustServiceError::InvalidChallenge)?;
        let len = usize::try_from(meta.length).map_err(|_| TrustServiceError::InvalidChallenge)?;
        let end = start.checked_add(len).ok_or(TrustServiceError::InvalidChallenge)?;
        let payload = container
            .get(start..end)
            .ok_or(TrustServiceError::InvalidChallenge)?;

        let dims_len = (meta.ndim as usize)
            .checked_mul(4)
            .ok_or(TrustServiceError::InvalidChallenge)?;
        let shape_end = meta
            .shape_offset
            .checked_add(dims_len)
            .ok_or(TrustServiceError::InvalidChallenge)?;
        let shape_bytes = handle
            .scratch
            .get(meta.shape_offset..shape_end)
            .ok_or(TrustServiceError::InvalidChallenge)?;
        let mut dims = [0u32, 0u32];
        for (k, chunk) in shape_bytes.chunks_exact(4).enumerate() {
            dims[k] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }

        records[records_count] = BlobDesc {
            name,
            dtype: meta.dtype,
            dims,
            ndim: meta.ndim,
            payload,
        };
        records_count += 1;
    }

    let existing_count = records_count;

    let mut dtype = 0u8;
    let mut layer_count = 0u32;
    let mut weights_len = 0u32;
    let mut biases_len = 0u32;
    let mut weights_bytes = 0u64;
    let mut biases_bytes = 0u64;
    for rec in &records[..existing_count] {
        match rec.name {
            BLOB_WEIGHTS => {
                dtype = rec.dtype;
                weights_len = rec.dims[0];
                weights_bytes = rec.payload.len() as u64;
            }
            BLOB_BIASES => {
                biases_len = rec.dims[0];
                biases_bytes = rec.payload.len() as u64;
            }
            BLOB_LAYER_META => {
                layer_count = rec.dims[0];
            }
            _ => {}
        }
    }

    let policy_bytes = [distribution_policy];
    records[records_count] = BlobDesc {
        name: AUTH_DISTRIBUTION_POLICY_BLOB_DATA,
        dtype: 0,
        dims: [1u32, 0],
        ndim: 1,
        payload: &policy_bytes,
    };
    records_count += 1;

    let mut message = CanonicalMessage::new();
    message.extend(&(bind_device.len() as u16).to_le_bytes())?;
    message.extend(bind_device)?;
    for rec in &records[..records_count] {
        if rec.name == AUTH_HMAC_SHA256_BLOB_DATA
            || rec.name == AUTH_ED25519_SIG_BLOB_DATA
            || rec.name == AUTH_ED25519_PUBKEY_BLOB_DATA
        {
            continue;
        }
        let name_bytes = rec.name.as_bytes();
        if name_bytes.len() > u16::MAX as usize {
            return Err(TrustServiceError::InvalidChallenge);
        }
        message.extend(&(name_bytes.len() as u16).to_le_bytes())?;
        message.extend(name_bytes)?;
        message.push(rec.dtype)?;
        message.push(rec.ndim)?;
        for k in 0..rec.ndim as usize {
            message.extend(&rec.dims[k].to_le_bytes())?;
        }
        message.extend(&(rec.payload.len() as u64).to_le_bytes())?;
        message.extend(rec.payload)?;
    }

    let signature = ed25519_sign(secret_key, message.as_slice());
    let public_key = ed25519_public_from_secret(secret_key);
    let metadata = create_encryption_metadata(1, 1, 1, [0u8; crate::security::crypto::RNN_ENCRYPTED_NONCE_SIZE]);
    let mut verified = [0u8; ASYM_SIGNATURE_LEN];
    verified.copy_from_slice(&signature);
    let _ = verify_signature(&public_key, message.as_slice(), &verified);
    let _ = metadata;

    records[records_count] = BlobDesc {
        name: AUTH_ED25519_SIG_BLOB_DATA,
        dtype: 0,
        dims: [ASYM_SIGNATURE_LEN as u32, 0],
        ndim: 1,
        payload: &signature,
    };
    records_count += 1;
    records[records_count] = BlobDesc {
        name: AUTH_ED25519_PUBKEY_BLOB_DATA,
        dtype: 0,
        dims: [ASYM_PUBLIC_KEY_LEN as u32, 0],
        ndim: 1,
        payload: &public_key,
    };
    records_count += 1;

    let model_name = header_tlv_payload(container, TLV_HEADER_MODEL_NAME)
        .ok()
        .and_then(|v| core::str::from_utf8(v).ok())
        .filter(|s| !s.is_empty());

    let precision: &str = if dtype == 0 { "f32" } else { "f64" };
    let total_params = u64::from(weights_len) + u64::from(biases_len);

    let mut bench_buf = [0u8; SIGN_BENCHMARK_CAP];
    let bmk_len = encode_distribution_benchmark(
        precision, layer_count, total_params, weights_bytes, biases_bytes, 0, &mut bench_buf,
    )?;

    let used = assemble_rnn_container(&records[..records_count], model_name, &bench_buf[..bmk_len], out)
        .map_err(|_| TrustServiceError::InvalidChallenge)?;

    let mut final_bench = [0u8; SIGN_BENCHMARK_CAP];
    let final_len = encode_distribution_benchmark(
        precision, layer_count, total_params, weights_bytes, biases_bytes, used, &mut final_bench,
    )?;
    let (bench_start, bench_end) =
        crate::format::model_config::rnn::parse_header_tlv_range(&out[..used], TLV_HEADER_BENCHMARK)
            .map_err(|_| TrustServiceError::InvalidChallenge)?;
    if bench_end - bench_start != final_len {
        return Err(TrustServiceError::InvalidChallenge);
    }
    out[bench_start..bench_end].copy_from_slice(&final_bench[..final_len]);

    Ok(used)
}

fn encode_distribution_benchmark(
    precision: &str,
    layer_count: u32,
    total_params: u64,
    weights_bytes: u64,
    biases_bytes: u64,
    output_bytes: usize,
    out: &mut [u8],
) -> Result<usize, TrustServiceError> {
    let metrics = BenchmarkMetrics {
        model_name: "",
        precision,
        elapsed_ms: 0,
        iterations: 0,
        train_samples: 0,
        avg_loss: 0.0,
        last_loss: 0.0,
        output_bytes,
        total_params,
        layer_count,
        input_dim: 0,
        output_dim: 0,
        benchmark_flags: 0,
        weights_bytes,
        biases_bytes,
        min_loss: 0.0,
        max_loss: 0.0,
        loss_stddev: 0.0,
        iterations_per_sec: 0.0,
        samples_per_sec: 0.0,
        eval_loss: 0.0,
        eval_accuracy: 0.0,
        eval_f1: 0.0,
        eval_mae: 0.0,
        eval_samples: 0,
        eval_dataset_hash: 0,
        logical_cores: 0,
        avg_frequency_mhz: 0,
        max_frequency_mhz: 0,
        max_workers: 0,
        target_cpu_utilization: 0.0,
    };
    encode_benchmark_blob(&metrics, out).map_err(|_| TrustServiceError::InvalidChallenge)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrustServiceError {
    InvalidChallenge,
    ServiceUnavailable,
    TransportNotConfigured,
    Unauthorized,
    InvalidDecision,
    TokenExpired,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActivationChallenge<'a> {
    pub model_sha256: [u8; 32],
    pub model_signing_pubkey: [u8; ASYM_PUBLIC_KEY_LEN],
    pub distribution_policy: u8,
    pub machine_fingerprint: &'a [u8],
    pub device_id: &'a [u8],
    pub issued_at_unix: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

pub fn encode_activation_challenge_binary(
    challenge: &ActivationChallenge<'_>,
    out: &mut [u8],
) -> Result<usize, TrustServiceError> {
    if challenge.machine_fingerprint.len() > u16::MAX as usize
        || challenge.device_id.len() > u16::MAX as usize
    {
        return Err(TrustServiceError::InvalidChallenge);
    }

    let needed = 4usize
        .checked_add(1)
        .and_then(|v| v.checked_add(32))
        .and_then(|v| v.checked_add(ASYM_PUBLIC_KEY_LEN))
        .and_then(|v| v.checked_add(1))
        .and_then(|v| v.checked_add(2))
        .and_then(|v| v.checked_add(challenge.machine_fingerprint.len()))
        .and_then(|v| v.checked_add(2))
        .and_then(|v| v.checked_add(challenge.device_id.len()))
        .and_then(|v| v.checked_add(8))
        .ok_or(TrustServiceError::InvalidChallenge)?;

    if out.len() < needed {
        return Err(TrustServiceError::InvalidChallenge);
    }

    let mut cursor = 0usize;
    out[cursor..cursor + 4].copy_from_slice(b"ATS0");
    cursor += 4;
    out[cursor] = 1;
    cursor += 1;

    out[cursor..cursor + 32].copy_from_slice(&challenge.model_sha256);
    cursor += 32;
    out[cursor..cursor + ASYM_PUBLIC_KEY_LEN].copy_from_slice(&challenge.model_signing_pubkey);
    cursor += ASYM_PUBLIC_KEY_LEN;
    out[cursor] = challenge.distribution_policy;
    cursor += 1;

    let mlen = challenge.machine_fingerprint.len() as u16;
    out[cursor..cursor + 2].copy_from_slice(&mlen.to_le_bytes());
    cursor += 2;
    out[cursor..cursor + challenge.machine_fingerprint.len()]
        .copy_from_slice(challenge.machine_fingerprint);
    cursor += challenge.machine_fingerprint.len();

    let dlen = challenge.device_id.len() as u16;
    out[cursor..cursor + 2].copy_from_slice(&dlen.to_le_bytes());
    cursor += 2;
    out[cursor..cursor + challenge.device_id.len()].copy_from_slice(challenge.device_id);
    cursor += challenge.device_id.len();

    out[cursor..cursor + 8].copy_from_slice(&challenge.issued_at_unix.to_le_bytes());
    cursor += 8;

    Ok(cursor)
}

pub fn validate_activation_decision_placeholder(
    decision: &ActivationDecision,
    now_unix: u64,
) -> Result<(), TrustServiceError> {
    if !decision.allowed {
        return Err(TrustServiceError::Unauthorized);
    }
    if decision.activation_token_len == 0
        || decision.activation_token_len > ACTIVATION_TOKEN_MAX_LEN
    {
        return Err(TrustServiceError::InvalidDecision);
    }
    if decision.expires_unix < now_unix {
        return Err(TrustServiceError::TokenExpired);
    }
    Ok(())
}

#[cfg(test)]
mod sign_tests {
    extern crate std;
    use super::*;
    use std::vec;

    #[test]
    fn sign_then_validate_device_locked_roundtrip() {
        let topology = [4usize, 3, 2];
        let (w_count, b_count) =
            crate::base::initializers::expected_parameter_counts(&topology).unwrap();
        let mut weights = vec![0.0f32; w_count];
        let mut biases = vec![0.0f32; b_count];
        let mut model = vec![0u8; 65536];

        let used = crate::api::rnn_api::build::build_f32_with_request(
            crate::api::rnn_api::core_api::BuildF32Request {
                model_name: "sign_roundtrip",
                topology: &topology,
                seed: Some(0xA17C),
                weights: &mut weights,
                biases: &mut biases,
                runtime_input: None,
                conv_spec: None,
                conv_in_shape: [0; 5],
                conv_layers: &[],
                benchmark_override: None,
                lm_params: &[],
                dataset_dirs: &[],
            },
            &mut model,
        )
        .unwrap();

        let plain_len = crate::security::crypto::decrypt_rnn(&mut model, used);

        let secret_key = [7u8; ASYM_SECRET_KEY_LEN];
        let device_id = b"device-0001";
        let mut signed = vec![0u8; 131072];
        let signed_len = sign_model_distribution(
            &model[..plain_len],
            &secret_key,
            device_id,
            DISTRIBUTION_POLICY_USER_DEVICE_LOCKED,
            &mut signed,
        )
        .unwrap();
        assert!(signed_len > 0 && signed_len <= signed.len());

        crate::format::model_config::validate_distribution_secure(&signed[..signed_len], device_id, &[])
            .unwrap();
    }

    #[test]
    fn tampered_signature_is_rejected() {
        let topology = [4usize, 3, 2];
        let (w_count, b_count) =
            crate::base::initializers::expected_parameter_counts(&topology).unwrap();
        let mut weights = vec![0.0f32; w_count];
        let mut biases = vec![0.0f32; b_count];
        let mut model = vec![0u8; 65536];

        let used = crate::api::rnn_api::build::build_f32_with_request(
            crate::api::rnn_api::core_api::BuildF32Request {
                model_name: "sign_tamper",
                topology: &topology,
                seed: Some(0xB22D),
                weights: &mut weights,
                biases: &mut biases,
                runtime_input: None,
                conv_spec: None,
                conv_in_shape: [0; 5],
                conv_layers: &[],
                benchmark_override: None,
                lm_params: &[],
                dataset_dirs: &[],
            },
            &mut model,
        )
        .unwrap();

        let plain_len = crate::security::crypto::decrypt_rnn(&mut model, used);

        let secret_key = [9u8; ASYM_SECRET_KEY_LEN];
        let device_id = b"device-0002";
        let mut signed = vec![0u8; 131072];
        let signed_len = sign_model_distribution(
            &model[..plain_len],
            &secret_key,
            device_id,
            DISTRIBUTION_POLICY_USER_DEVICE_LOCKED,
            &mut signed,
        )
        .unwrap();

        crate::format::model_config::validate_distribution_secure(&signed[..signed_len], device_id, &[])
            .unwrap();

        let wrong_device = b"device-XXXX";
        assert!(crate::format::model_config::validate_distribution_secure(
            &signed[..signed_len],
            wrong_device,
            &[]
        )
        .is_err());
    }
}

pub mod trained {
    pub const SMALL_F32_PATH: &str = "trained/f32/small.rnn";
    pub const SMALL_F64_PATH: &str = "trained/f64/small.rnn";
    pub const MEDIUM_F32_PATH: &str = "trained/f32/medium.rnn";
    pub const MEDIUM_F64_PATH: &str = "trained/f64/medium.rnn";
    pub const LARGE_F32_PATH: &str = "trained/f32/large.rnn";
    pub const LARGE_F64_PATH: &str = "trained/f64/large.rnn";
    pub const ENORMOUS_F32_PATH: &str = "trained/f32/enormous.rnn";
    pub const ENORMOUS_F64_PATH: &str = "trained/f64/enormous.rnn";

    pub fn rename(buf: &mut [u8], name: &str) -> bool {
        if buf.len() < 0x20 {
            return false;
        }
        let name_bytes = name.as_bytes();
        let len = name_bytes.len().min(16);
        buf[0x10..0x20].fill(0);
        buf[0x10..0x10 + len].copy_from_slice(&name_bytes[..len]);
        true
    }
}
