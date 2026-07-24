use core::convert::TryInto;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchmarkMetrics<'a> {
    pub model_name: &'a str,
    pub precision: &'a str,
    pub elapsed_ms: u64,
    pub iterations: u64,
    pub train_samples: u64,
    pub avg_loss: f32,
    pub last_loss: f32,
    pub output_bytes: usize,
    pub total_params: u64,
    pub layer_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub benchmark_flags: u64,
    pub weights_bytes: u64,
    pub biases_bytes: u64,
    pub min_loss: f32,
    pub max_loss: f32,
    pub loss_stddev: f32,
    pub iterations_per_sec: f32,
    pub samples_per_sec: f32,
    pub eval_loss: f32,
    pub eval_accuracy: f32,
    pub eval_f1: f32,
    pub eval_mae: f32,
    pub eval_samples: u64,
    pub eval_dataset_hash: u64,
    pub logical_cores: u32,
    pub avg_frequency_mhz: u32,
    pub max_frequency_mhz: u32,
    pub max_workers: u32,
    pub target_cpu_utilization: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchmarkMetricsView<'a> {
    pub model_name: &'a str,
    pub precision: &'a str,
    pub elapsed_ms: u64,
    pub iterations: u64,
    pub train_samples: u64,
    pub avg_loss: f32,
    pub last_loss: f32,
    pub output_bytes: usize,
    pub total_params: u64,
    pub layer_count: u32,
    pub input_dim: u32,
    pub output_dim: u32,
    pub benchmark_flags: u64,
    pub weights_bytes: u64,
    pub biases_bytes: u64,
    pub min_loss: f32,
    pub max_loss: f32,
    pub loss_stddev: f32,
    pub iterations_per_sec: f32,
    pub samples_per_sec: f32,
    pub eval_loss: f32,
    pub eval_accuracy: f32,
    pub eval_f1: f32,
    pub eval_mae: f32,
    pub eval_samples: u64,
    pub eval_dataset_hash: u64,
    pub logical_cores: u32,
    pub avg_frequency_mhz: u32,
    pub max_frequency_mhz: u32,
    pub max_workers: u32,
    pub target_cpu_utilization: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BenchmarkEncodeError {
    BufferTooSmall,
    InvalidFormat,
}

const BENCHMARK_MAGIC: [u8; 4] = [b'B', b'M', b'K', 0x01];
const BENCHMARK_VERSION: u16 = 4;
const BENCHMARK_HEADER_SIZE: usize =
    4 + 2 + 2 + 8 + 8 + 8 + 4 + 4 + 8 + 8 + 4 + 4 + 4 + 8 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 8 + 8 + 4 + 4 + 4 + 4 + 4 + 2 + 2;

pub fn encoded_size_benchmark_blob(metrics: &BenchmarkMetrics<'_>) -> Option<usize> {
    BENCHMARK_HEADER_SIZE
        .checked_add(metrics.model_name.len())?
        .checked_add(metrics.precision.len())
}

pub fn encode_benchmark_blob(
    metrics: &BenchmarkMetrics<'_>,
    out: &mut [u8],
) -> Result<usize, BenchmarkEncodeError> {
    if metrics.model_name.len() > u16::MAX as usize || metrics.precision.len() > u16::MAX as usize {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }
    let needed = encoded_size_benchmark_blob(metrics).ok_or(BenchmarkEncodeError::InvalidFormat)?;
    if out.len() < needed {
        return Err(BenchmarkEncodeError::BufferTooSmall);
    }

    out[0..4].copy_from_slice(&BENCHMARK_MAGIC);
    out[4..6].copy_from_slice(&BENCHMARK_VERSION.to_le_bytes());
    out[6..8].copy_from_slice(&0u16.to_le_bytes());
    out[8..16].copy_from_slice(&metrics.elapsed_ms.to_le_bytes());
    out[16..24].copy_from_slice(&metrics.iterations.to_le_bytes());
    out[24..32].copy_from_slice(&metrics.train_samples.to_le_bytes());
    out[32..36].copy_from_slice(&metrics.avg_loss.to_le_bytes());
    out[36..40].copy_from_slice(&metrics.last_loss.to_le_bytes());
    out[40..48].copy_from_slice(&(metrics.output_bytes as u64).to_le_bytes());
    out[48..56].copy_from_slice(&metrics.total_params.to_le_bytes());
    out[56..60].copy_from_slice(&metrics.layer_count.to_le_bytes());
    out[60..64].copy_from_slice(&metrics.input_dim.to_le_bytes());
    out[64..68].copy_from_slice(&metrics.output_dim.to_le_bytes());
    out[68..76].copy_from_slice(&metrics.benchmark_flags.to_le_bytes());
    out[76..84].copy_from_slice(&metrics.weights_bytes.to_le_bytes());
    out[84..92].copy_from_slice(&metrics.biases_bytes.to_le_bytes());
    out[92..96].copy_from_slice(&metrics.min_loss.to_le_bytes());
    out[96..100].copy_from_slice(&metrics.max_loss.to_le_bytes());
    out[100..104].copy_from_slice(&metrics.loss_stddev.to_le_bytes());
    out[104..108].copy_from_slice(&metrics.iterations_per_sec.to_le_bytes());
    out[108..112].copy_from_slice(&metrics.samples_per_sec.to_le_bytes());
    out[112..116].copy_from_slice(&metrics.eval_loss.to_le_bytes());
    out[116..120].copy_from_slice(&metrics.eval_accuracy.to_le_bytes());
    out[120..124].copy_from_slice(&metrics.eval_f1.to_le_bytes());
    out[124..128].copy_from_slice(&metrics.eval_mae.to_le_bytes());
    out[128..136].copy_from_slice(&metrics.eval_samples.to_le_bytes());
    out[136..144].copy_from_slice(&metrics.eval_dataset_hash.to_le_bytes());
    out[144..148].copy_from_slice(&metrics.logical_cores.to_le_bytes());
    out[148..152].copy_from_slice(&metrics.avg_frequency_mhz.to_le_bytes());
    out[152..156].copy_from_slice(&metrics.max_frequency_mhz.to_le_bytes());
    out[156..160].copy_from_slice(&metrics.max_workers.to_le_bytes());
    out[160..164].copy_from_slice(&metrics.target_cpu_utilization.to_le_bytes());
    out[164..166].copy_from_slice(&(metrics.model_name.len() as u16).to_le_bytes());
    out[166..168].copy_from_slice(&(metrics.precision.len() as u16).to_le_bytes());

    let mut cursor = BENCHMARK_HEADER_SIZE;
    let model_bytes = metrics.model_name.as_bytes();
    out[cursor..cursor + model_bytes.len()].copy_from_slice(model_bytes);
    cursor += model_bytes.len();

    let precision_bytes = metrics.precision.as_bytes();
    out[cursor..cursor + precision_bytes.len()].copy_from_slice(precision_bytes);
    cursor += precision_bytes.len();

    Ok(cursor)
}

pub fn decode_benchmark_blob<'a>(
    bytes: &'a [u8],
) -> Result<BenchmarkMetricsView<'a>, BenchmarkEncodeError> {
    if bytes.len() < BENCHMARK_HEADER_SIZE {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }
    if bytes[0..4] != BENCHMARK_MAGIC {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != BENCHMARK_VERSION {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }

    let elapsed_ms = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let iterations = u64::from_le_bytes(
        bytes[16..24]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let train_samples = u64::from_le_bytes(
        bytes[24..32]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let avg_loss = f32::from_le_bytes(
        bytes[32..36]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let last_loss = f32::from_le_bytes(
        bytes[36..40]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let output_bytes = u64::from_le_bytes(
        bytes[40..48]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    ) as usize;
    let total_params = u64::from_le_bytes(
        bytes[48..56]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let layer_count = u32::from_le_bytes(
        bytes[56..60]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let input_dim = u32::from_le_bytes(
        bytes[60..64]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let output_dim = u32::from_le_bytes(
        bytes[64..68]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let benchmark_flags = u64::from_le_bytes(
        bytes[68..76]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let weights_bytes = u64::from_le_bytes(
        bytes[76..84]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let biases_bytes = u64::from_le_bytes(
        bytes[84..92]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let min_loss = f32::from_le_bytes(
        bytes[92..96]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let max_loss = f32::from_le_bytes(
        bytes[96..100]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let loss_stddev = f32::from_le_bytes(
        bytes[100..104]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let iterations_per_sec = f32::from_le_bytes(
        bytes[104..108]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let samples_per_sec = f32::from_le_bytes(
        bytes[108..112]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let eval_loss = f32::from_le_bytes(
        bytes[112..116]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let eval_accuracy = f32::from_le_bytes(
        bytes[116..120]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let eval_f1 = f32::from_le_bytes(
        bytes[120..124]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let eval_mae = f32::from_le_bytes(
        bytes[124..128]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let eval_samples = u64::from_le_bytes(
        bytes[128..136]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let eval_dataset_hash = u64::from_le_bytes(
        bytes[136..144]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let logical_cores = u32::from_le_bytes(
        bytes[144..148]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let avg_frequency_mhz = u32::from_le_bytes(
        bytes[148..152]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let max_frequency_mhz = u32::from_le_bytes(
        bytes[152..156]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let max_workers = u32::from_le_bytes(
        bytes[156..160]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let target_cpu_utilization = f32::from_le_bytes(
        bytes[160..164]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let model_len = u16::from_le_bytes(
        bytes[164..166]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    ) as usize;
    let precision_len = u16::from_le_bytes(
        bytes[166..168]
            .try_into()
            .map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    ) as usize;

    let model_end = BENCHMARK_HEADER_SIZE
        .checked_add(model_len)
        .ok_or(BenchmarkEncodeError::InvalidFormat)?;
    let precision_end = model_end
        .checked_add(precision_len)
        .ok_or(BenchmarkEncodeError::InvalidFormat)?;
    if precision_end > bytes.len() {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }

    let model_name = core::str::from_utf8(&bytes[BENCHMARK_HEADER_SIZE..model_end])
        .map_err(|_| BenchmarkEncodeError::InvalidFormat)?;
    let precision = core::str::from_utf8(&bytes[model_end..precision_end])
        .map_err(|_| BenchmarkEncodeError::InvalidFormat)?;

    Ok(BenchmarkMetricsView {
        model_name,
        precision,
        elapsed_ms,
        iterations,
        train_samples,
        avg_loss,
        last_loss,
        output_bytes,
        total_params,
        layer_count,
        input_dim,
        output_dim,
        benchmark_flags,
        weights_bytes,
        biases_bytes,
        min_loss,
        max_loss,
        loss_stddev,
        iterations_per_sec,
        samples_per_sec,
        eval_loss,
        eval_accuracy,
        eval_f1,
        eval_mae,
        eval_samples,
        eval_dataset_hash,
        logical_cores,
        avg_frequency_mhz,
        max_frequency_mhz,
        max_workers,
        target_cpu_utilization,
    })
}

fn decode_compact_benchmark_blob(bytes: &[u8]) -> Result<BenchmarkMetricsView<'_>, BenchmarkEncodeError> {
    if bytes.len() < 32 {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }
    if bytes[0..4] != BENCHMARK_MAGIC {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != BENCHMARK_VERSION {
        return Err(BenchmarkEncodeError::InvalidFormat);
    }
    let layer_count = bytes[7] as u32;
    let input_dim = u32::from_le_bytes(
        bytes[8..12].try_into().map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let output_dim = u32::from_le_bytes(
        bytes[12..16].try_into().map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let total_params = u32::from_le_bytes(
        bytes[16..20].try_into().map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    ) as u64;
    let _total_params_dup = u32::from_le_bytes(
        bytes[20..24].try_into().map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let last_loss = f32::from_le_bytes(
        bytes[24..28].try_into().map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    );
    let iterations = u32::from_le_bytes(
        bytes[28..32].try_into().map_err(|_| BenchmarkEncodeError::InvalidFormat)?,
    ) as u64;

    Ok(BenchmarkMetricsView {
        model_name: "",
        precision: "",
        elapsed_ms: 0,
        iterations,
        train_samples: 0,
        avg_loss: 0.0,
        last_loss,
        output_bytes: 0,
        total_params,
        layer_count,
        input_dim,
        output_dim,
        benchmark_flags: 0,
        weights_bytes: 0,
        biases_bytes: 0,
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
    })
}

pub fn get_bmk(rnn_bytes: &[u8]) -> Result<BenchmarkMetricsView<'_>, BenchmarkEncodeError> {
    let blob = get_bmk_raw(rnn_bytes)?;
    decode_benchmark_blob(blob).or_else(|_| decode_compact_benchmark_blob(blob))
}

pub fn get_bmk_raw(rnn_bytes: &[u8]) -> Result<&[u8], BenchmarkEncodeError> {
    let blob = crate::format::model_format::header_tlv_payload(
        rnn_bytes,
        crate::format::model_format::TLV_HEADER_BENCHMARK,
    )
    .map_err(|_| BenchmarkEncodeError::InvalidFormat)?;
    Ok(blob)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrainingSummary {
    pub elapsed_ms: u64,
    pub iterations: u64,
    pub train_samples: u64,
    pub avg_loss: f32,
    pub last_loss: f32,
    pub min_loss: f32,
    pub max_loss: f32,
    pub loss_stddev: f32,
    pub iterations_per_sec: f32,
    pub samples_per_sec: f32,
    pub logical_cores: u32,
    pub max_workers: u32,
    pub target_cpu_utilization: f32,
    pub avg_frequency_mhz: u32,
    pub max_frequency_mhz: u32,
}

pub fn patch_benchmark_blob_with_training(
    existing: &[u8],
    summary: &TrainingSummary,
    out: &mut [u8],
) -> Result<usize, BenchmarkEncodeError> {
    let view = decode_benchmark_blob(existing)?;
    let metrics = BenchmarkMetrics {
        model_name: view.model_name,
        precision: view.precision,
        elapsed_ms: summary.elapsed_ms,
        iterations: summary.iterations,
        train_samples: summary.train_samples,
        avg_loss: summary.avg_loss,
        last_loss: summary.last_loss,
        output_bytes: view.output_bytes,
        total_params: view.total_params,
        layer_count: view.layer_count,
        input_dim: view.input_dim,
        output_dim: view.output_dim,
        benchmark_flags: view.benchmark_flags,
        weights_bytes: view.weights_bytes,
        biases_bytes: view.biases_bytes,
        min_loss: summary.min_loss,
        max_loss: summary.max_loss,
        loss_stddev: summary.loss_stddev,
        iterations_per_sec: summary.iterations_per_sec,
        samples_per_sec: summary.samples_per_sec,
        eval_loss: view.eval_loss,
        eval_accuracy: view.eval_accuracy,
        eval_f1: view.eval_f1,
        eval_mae: view.eval_mae,
        eval_samples: view.eval_samples,
        eval_dataset_hash: view.eval_dataset_hash,
        logical_cores: summary.logical_cores,
        avg_frequency_mhz: summary.avg_frequency_mhz,
        max_frequency_mhz: summary.max_frequency_mhz,
        max_workers: summary.max_workers,
        target_cpu_utilization: summary.target_cpu_utilization,
    };
    encode_benchmark_blob(&metrics, out)
}

