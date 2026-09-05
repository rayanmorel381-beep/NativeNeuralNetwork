//! Extraction of the raw benchmark block from an RNN container.

use nnn_bmk::{BenchmarkEncodeError, BenchmarkMetrics};

pub fn get_bmk_raw(rnn_bytes: &[u8]) -> Result<&[u8], BenchmarkEncodeError> {
    nnn_bmk::get_bmk_raw(rnn_bytes)
}

pub(crate) fn parser_metrics<'a>(
    model_name: &'a str,
    precision: &'a str,
    elapsed_ns: u64,
    output_bytes: usize,
) -> BenchmarkMetrics<'a> {
    BenchmarkMetrics {
        model_name,
        precision,
        elapsed_ms: elapsed_ns / 1_000_000,
        iterations: 1,
        train_samples: 0,
        avg_loss: 0.0,
        last_loss: 0.0,
        output_bytes,
        total_params: 0,
        layer_count: 0,
        input_dim: 0,
        output_dim: 0,
        benchmark_flags: 0,
        weights_bytes: 0,
        biases_bytes: 0,
        min_loss: 0.0,
        max_loss: 0.0,
        loss_stddev: 0.0,
        iterations_per_sec: if elapsed_ns == 0 { 0.0 } else { 1_000_000_000.0 / elapsed_ns as f32 },
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
    }
}