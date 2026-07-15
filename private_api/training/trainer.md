# training/trainer

The training loop and everything around it: SGD/AdamW steps, the fork-based
worker pool, checkpointing and the resource policy that throttles CPU. The public
`train` entry point calls `train_lm` here.

Façade: `private_api::modules::trainer`.

## Key types

- `TrainLoopConfig` / `TrainLoopResult` — loop configuration and outcome.
- `SgdConfig` / `SgdConfigF64` / `SgdScratch` / `SgdScratchF64` — SGD state.
- `AdamState` — AdamW moment state.
- `ApplySgdArgs` / `ApplySgdArgsF64` — one-step argument bundles.
- `TrainBatchedBuffers` / `TrainScratch` — reusable batched buffers.
- `LmTrainConfig` / `LmTrainMode` / `LmTrainRun` / `LmTrainStep` / `LmTrainBufs`
  — language-model training path.
- `ParsedLayout` — parsed parameter layout.
- `TrainError` — failure reason.

## Representative functions

- Entry: `train`, `train_lm`, `run_lm_train`, `train_loop_with_policy`.
- Steps: `train_step`, `train_step_masked`, `sgd_step`, `sgd_step_f64`,
  `apply_grads_adamw`, `train_batched`.
- Grads: `accumulate_grads`, `accumulate_grads_masked`.
- Forward: `forward_loss_batched`, `forward_predict_batched`.
- Sizing/validation: `required_train_buffer_len`, `batched_work_count`,
  `block_param_count`, `config_is_valid`, `io_matches_layers`,
  `effective_cycle_size`, `checkpoint_count`.
- Schedules: `with_warmup`, `default_adam`, `warm_start`.

## Concrete usage

The whole training call is a single line — this is exactly what the `training/`
binaries run after `build`:

```rust
use native_neural_network::rnn_api::train;

// `model_bytes` came from build_f32/build_f64.
let trained = train(&mut model_bytes[..built], /* seconds */ 1800, &mut out)
    .expect("train failed");
// out[..trained] is a fully-formed, resumable .rnn container.
```

`train_loop_with_policy` is where the CPU budget matters: the policy computes an
`active_workers` count and the fork pool is started/finished with exactly that
many workers, so training never saturates all cores.

## Performance levers

- Training respects the [runtime](../runtime/runtime.md) `ConsumptionGuard`:
  workers are clamped so CPU stays ≤ 80 % and RAM ≤ 70 % — this is what keeps
  Veyma responsive on a 32 GB box.
- Fork workers block on a futex when idle → **0 %** CPU while waiting.
- Re-running `train` resumes from the latest snapshot (checkpointing).
- Prefer AdamW (`default_adam` + `apply_grads_adamw`) for stability; SGD for raw
  speed.

## Integration

Optimizes [lm](../transformer/lm.md) / [network](../core/network.md) params using
[optimizers](../training/optimizers.md), [losses](../training/losses.md),
[gradients](../training/gradients.md), [schedulers](../training/schedulers.md);
writes results via [model_format](../core/model_format.md) and
[benchmark](../runtime/benchmark.md).
