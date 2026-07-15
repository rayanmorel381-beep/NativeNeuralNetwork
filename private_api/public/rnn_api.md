# public/rnn_api

The **stable public API** — the only supported entry points. Everything under
`private_api::modules::*` exists to build on top of these five functions.

Path: `native_neural_network::rnn_api`, re-exported at crate root and via
`private_api::{build_f32, build_f64, train, run, read_rnn}`.

## Types

- `BuildF32Request<'a>` / `BuildF64Request<'a>` — full build requests (topology,
  seed, weights/biases, LM params, dataset dirs, optional conv spec / runtime
  input / benchmark override).
- `RnnApiError` — the single error type returned by every entry point.

## Functions

- `build_f32(...)` / `build_f64(...)` — convenience builders (positional args).
- `build_f32_with_request(...)` / `build_f64_with_request(...)` — full-control
  builders taking a `Build*Request`.
- `train(model_bytes, seconds, out) -> usize` — train for N seconds; returns the
  trained container length.
- `run(bytes, config) -> (f64, usize)` — inference; returns `(score, out_len)`.
- `read_rnn(bytes, device_id) -> ReadableRnnFormat` — parse & validate a
  container (optional `device_id` for owner-bound decryption).

## Concrete usage — full-control build

The convenience `build_f32` covers the common case; use a request when you need
a conv spec, a runtime input, or a benchmark override:

```rust
use native_neural_network::rnn_api::{build_f32_with_request, BuildF32Request};

let mut weights = [0.0f32; 73_728];
let mut biases = [0.0f32; 448];
let mut out = vec![0u8; 2 << 30];

let req = BuildF32Request {
    model_name: "m.rnn",
    topology: &[128, 256, 128, 64],
    seed: Some(1),
    weights: &mut weights,
    biases: &mut biases,
    runtime_input: None,
    conv_spec: None,
    conv_in_shape: [0; 5],
    conv_layers: &[],
    benchmark_override: None,
    lm_params: &[32768, 1024, 2048, 4096, 16, 16, 8, 128, 16384, 512],
    dataset_dirs: &["dataset/en", "dataset/fr"],
};

let n = build_f32_with_request(req, &mut out).expect("build failed");
```

## Contract

- All output buffers are **caller-owned**; no hidden allocation on the hot path.
- Every function returns `Result<_, RnnApiError>` — handle it, never unwrap in
  production.
- `read_rnn` is the only sanctioned way to inspect a container (see
  [core/model_format](../core/model_format.md)).

## Integration

Drives [trainer](../training/trainer.md) (`train`),
[inference](../transformer/inference.md) (`run`) and
[model_format](../core/model_format.md) (`read_rnn`).
