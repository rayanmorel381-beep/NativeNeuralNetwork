# runtime/benchmark

The training-metrics blob stored inside every `.rnn`. It records loss curves,
throughput, evaluation metrics and hardware context, and can be patched after a
training run.

Façade: `private_api::modules::benchmark`.

## Key types

- `BenchmarkMetrics` / `BenchmarkMetricsView` — the metric record and a read view.
- `TrainingSummary` — the per-run summary patched into the blob.
- `BenchmarkEncodeError` — failure reason.

## Functions

- `encode_benchmark_blob(...)` / `decode_benchmark_blob(...)` /
  `encoded_size_benchmark_blob(...)` — blob round-trip.
- `patch_benchmark_blob_with_training(...)` — fold a run's summary into the blob.
- `get_bmk(...)` / `get_bmk_raw(...)` — extract the blob from a container.

## Concrete usage

Read the benchmark metrics out of a trained `.rnn` (this is what the training
binaries do to emit their `.csv`/`.json`/`.yaml` exports):

```rust
use native_neural_network::rnn_api::read_rnn;

fn dump_metrics(rnn_bytes: &mut [u8]) {
    let view = read_rnn(rnn_bytes, None).expect("read failed");
    // The view exposes the benchmark fields directly:
    //   view.avg_loss, view.last_loss, view.iterations_per_sec,
    //   view.eval_accuracy, view.logical_cores, view.max_workers, ...
    let _ = view;
}
```

## Performance levers

- The blob is small and versioned — reading metrics never requires decoding the
  full model.
- Use `patch_benchmark_blob_with_training` to append a run's summary so history
  grows across successive `train` calls (resumable benchmarking).

## Integration

Written into the [model_format](../core/model_format.md) container by
[trainer](../training/trainer.md); metrics come from
[metrics](../numerics/metrics.md); throughput pairs with
[profiler](../runtime/profiler.md).
