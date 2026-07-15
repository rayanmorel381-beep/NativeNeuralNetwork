# runtime/profiler

A lightweight op counter for roofline-style analysis: count matmuls,
activations, attention, normalization and memory traffic, then derive
throughput and arithmetic intensity.

Façade: `private_api::modules::profiler`.

## Key types

- `OpCounter` — the accumulator you feed events into.

## Functions

- Record: `add_matmul`, `add_activation`, `add_attention`, `add_normalization`,
  `add_memory_read`, `add_memory_write`.
- Derive: `total_ops`, `total_memory_bytes`, `arithmetic_intensity`,
  `ops_per_second`, `bytes_per_second`, `is_memory_heavy`.
- Manage: `merge`, `reset`, `has_recorded_work`.

## Concrete usage

Instrument a forward pass, then decide whether it's compute- or memory-bound:

```rust
use native_neural_network::private_api::modules::profiler::OpCounter;

let mut ops = OpCounter::default();

fn record_matmul(ops: &mut OpCounter, m: usize, n: usize, k: usize) {
    ops.add_matmul(m, n, k);
    ops.add_memory_read((m * k + k * n) * 4);   // f32 bytes read
    ops.add_memory_write(m * n * 4);
}

// after the pass:
fn verdict(ops: &OpCounter, seconds: f64) {
    let intensity = ops.arithmetic_intensity();  // FLOP per byte
    let bound = if ops.is_memory_heavy() { "memory-bound" } else { "compute-bound" };
    let _ = (intensity, bound, ops.ops_per_second(seconds));
}
```

## Performance levers

- `arithmetic_intensity` tells you whether to optimize compute or memory —
  don't guess.
- `merge` per-worker counters after a parallel section instead of sharing one.

## Integration

Complements the persisted metrics in [benchmark](../runtime/benchmark.md);
instruments [inference](../transformer/inference.md) and
[trainer](../training/trainer.md).
