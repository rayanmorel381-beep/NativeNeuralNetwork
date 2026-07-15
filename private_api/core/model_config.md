# core/model_config

High-level model configuration and the resource policy that governs training.
`ModelConfig` is also the type `run` mutates during inference.

Façade: `private_api::modules::model_config`.

## Key types

- `ModelConfig` — the full model description; passed to `run`, produced by build.
- `Precision` — f32 vs f64 selection for the model.
- `ResourcePolicy` — CPU/RAM budget and parallelism policy.
- `TrainingMetrics` — running training statistics.
- `ProgressCallback` / `ProgressUpdate` — hooks to observe long-running work.
- `SampleFiller` — supplies training samples on demand.

## Concrete usage

Run inference: `run` takes the model bytes plus a mutable `ModelConfig` and
returns `(score, produced_len)`:

```rust
use native_neural_network::rnn_api::run;
use native_neural_network::model_config::ModelConfig;

fn infer(model_bytes: &mut [u8], config: &mut ModelConfig) {
    let (score, out_len) = run(model_bytes, config).expect("run failed");
    // `score` is the model's scalar output; `out_len` bytes were produced.
    let _ = (score, out_len);
}
```

## Performance levers

- Set `ResourcePolicy` so the [trainer](../training/trainer.md) worker pool is
  clamped by the [runtime](../runtime/runtime.md) `ConsumptionGuard`
  (CPU ≤ 80 %, RAM ≤ 70 %) — Veyma's stability on 32 GB relies on this.
- Choose `Precision` once; f32 halves memory and roughly doubles throughput vs
  f64 for the same topology.
- Use `ProgressCallback` to stream metrics without blocking training.

## Integration

Entry point for the public `build_*` / `run` API; drives
[trainer](../training/trainer.md) and is serialized into the
[model_format](../core/model_format.md) container.
