# numerics/metrics

Evaluation metrics and online accumulators: accuracy, MSE/MAE, cross-entropy from
probabilities, argmax, and running means.

Façade: `private_api::modules::metrics`.

## Key types

- `RunningMean` / `RunningMeanF64` — streaming mean accumulators (`update`,
  `merge`, `value`).
- `MetricError` — failure reason.

## Functions

- Classification: `accuracy_top1_from_one_hot_f32`, `..._f64`, `argmax_f32`,
  `argmax_f64`, `cross_entropy_from_probabilities_f32`, `..._f64`.
- Regression: `mse_f32`, `mse_f64`, `mae_f32`, `mae_f64`.

## Concrete usage

Track evaluation accuracy across a stream of batches without storing them:

```rust
use native_neural_network::private_api::modules::metrics::RunningMean;

let mut acc = RunningMean::default();

fn eval_batch(acc: &mut RunningMean, probs: &[f32], target: usize) {
    use native_neural_network::private_api::modules::metrics;
    let correct = metrics::argmax_f32(probs) == target;
    acc.update(if correct { 1.0 } else { 0.0 });
}

// later:
// let final_accuracy = acc.value();
```

The `.rnn` benchmark blob stores exactly these figures (`eval_accuracy`,
`eval_mae`, …) — see [benchmark](../runtime/benchmark.md).

## Performance levers

- `RunningMean` is O(1) memory — merge partials from parallel workers with
  `merge` instead of concatenating buffers.

## Integration

Computed by [trainer](../training/trainer.md) evaluation, persisted via
[benchmark](../runtime/benchmark.md).
