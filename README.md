# private_api — Internal module documentation

> **Who is this for?**
> A human who wants to understand *everything* the `native_neural_network` crate
> can do and push it to its limits. Every page documents a module that actually
> exists in `src/`, with **concrete, copy-pasteable usage examples**.
> This folder is documentation only — it is not compiled.

Veyma already drives this whole surface in production on a 32 GB machine: it
builds models, trains them under a CPU/RAM budget, quantizes them, runs chat and
text generation, and reads/validates `.rnn` containers — all through the same
modules documented here.

## The two layers

The crate is `#![no_std]`, **zero external dependencies** (only `std` inside the
training binaries). It exposes:

- A tiny **stable public API** (`build_f32`, `build_f64`, `train`, `run`,
  `read_rnn`) — the supported surface.
- A large set of **internal modules** (`pub(crate)` in [`src/lib.rs`](../src/lib.rs)),
  exposed as a navigable façade through `private_api::modules::*`. These are the
  building blocks; compose them, but never hand-craft an `.rnn` — always go
  through `model_format` / `read_rnn` so validation and crypto stay intact.

## Verified end-to-end example

This is exactly what the training binaries in [`training/`](../training) do
(`build → train → read`). It is the canonical "hello world" for the crate:

```rust
use native_neural_network::rnn_api::{build_f32, train, read_rnn};

// Feed-forward topology + language-model hyperparameters.
const TOPOLOGY: &[usize] = &[128, 256, 128, 64];
const LM_PARAMS: &[usize] = &[32768, 1024, 2048, 4096, 16, 16, 8, 128, 16384, 512];

const WEIGHTS_CAP: usize = 73_728;
const BIASES_CAP: usize = 448;

fn main() {
    let mut weights = [0.0f32; WEIGHTS_CAP];
    let mut biases = [0.0f32; BIASES_CAP];

    // Output buffers are caller-owned (no hidden allocation on the hot path).
    let mut model_buf = vec![0u8; 2 * 1024 * 1024 * 1024];
    let mut train_out = vec![0u8; 2 * 1024 * 1024 * 1024];

    // 1) BUILD — assemble a fresh model into `model_buf`.
    let built = build_f32(
        "small.rnn",
        TOPOLOGY,
        0xA1B2_C3D4_E5F6_0001u64,     // seed
        &mut weights,
        &mut biases,
        LM_PARAMS,
        &["dataset/en", "dataset/fr", "."],
        &mut model_buf,
    )
    .expect("build failed");

    // 2) TRAIN — run for N seconds, writing the trained container to `train_out`.
    let trained = train(&mut model_buf[..built], 1800, &mut train_out)
        .expect("train failed");

    // 3) READ — parse & validate the produced `.rnn` bytes.
    let view = read_rnn(&mut train_out[..trained], None)
        .expect("read failed");
    // `view` exposes benchmark metrics, param counts, etc.
}
```

`build_f64` / f64 buffers give the double-precision path; everything else is
identical.

For recipes that **combine several modules** (chat generation, quantization,
LoRA, integrity checks, roofline profiling…), see [COOKBOOK.md](COOKBOOK.md).

## Module map

### [public/](public) — the stable, supported surface
- [public/rnn_api](public/rnn_api.md) — `build_f32/f64`, `train`, `run`, `read_rnn` (+ request/error types)
- [public/ffi](public/ffi.md) — the C ABI (5 `extern "C"` symbols) behind every wrapper
- [public/trust_service](public/trust_service.md) — model signing & activation (feature-gated)

### [core/](core) — model core & container format
- [core/engine](core/engine.md) — backend dispatch (CPU/GPU/TPU/LPU), kernel contracts
- [core/network](core/network.md) — `NeuralNetwork` structure, stats, assembly
- [core/layers](core/layers.md) — layer description/validation/planning
- [core/model_config](core/model_config.md) — model configuration & resource policy
- [core/model_format](core/model_format.md) — RMD1/`.rnn` container encode/decode
- [core/rnn_format](core/rnn_format.md) — low-level `.rnn` scanner/validator (internal)
- [core/tensor](core/tensor.md) — tensor families (dense, quantized, sparse, GPU…)
- [core/scratch](core/scratch.md) — aligned memory arena, no dynamic allocation

### [transformer/](transformer) — LM block & decoding
- [transformer/lm](transformer/lm.md) — full language model (weights, config, chat, quant)
- [transformer/inference](transformer/inference.md) — forward/backward, generation, runtime LoRA
- [transformer/attention](transformer/attention.md) — SDPA, causal/sparse masks, stable softmax
- [transformer/kv_cache](transformer/kv_cache.md) — key/value cache for incremental generation
- [transformer/rope](transformer/rope.md) — rotary position embeddings
- [transformer/embeddings](transformer/embeddings.md) — embedding table & MLP head
- [transformer/normalization](transformer/normalization.md) — RMSNorm
- [transformer/moe](transformer/moe.md) — mixture-of-experts (top-1/top-2 routing)
- [transformer/beam_search](transformer/beam_search.md) — beam search decoding
- [transformer/sampling](transformer/sampling.md) — top-k, top-p, temperature, argmax
- [transformer/tokenizer](transformer/tokenizer.md) — BPE, vocab/merges, encode/decode

### [training/](training) — optimization
- [training/trainer](training/trainer.md) — training loop, SGD/AdamW, fork pool, policy
- [training/optimizers](training/optimizers.md) — AdamW/SGD, state, steps
- [training/gradients](training/gradients.md) — clipping, norms, NaN/Inf guards
- [training/losses](training/losses.md) — cross-entropy and gradients
- [training/schedulers](training/schedulers.md) — learning-rate schedules
- [training/initializers](training/initializers.md) — parameter initialization
- [training/lora](training/lora.md) — LoRA adaptation (low-rank delta)
- [training/conv_net](training/conv_net.md) — convolutional training parameters

### [geometry/](geometry) — convolutions & spatial
- [geometry/conv3d](geometry/conv3d.md) — 3D convolution
- [geometry/conv5d](geometry/conv5d.md) — 5D convolution
- [geometry/sphere5d](geometry/sphere5d.md) — 5D spherical neuron representation

### [numerics/](numerics) — primitives
- [numerics/math](numerics/math.md) — `no_std` transcendental functions
- [numerics/activations](numerics/activations.md) — activation functions & derivatives
- [numerics/quantization](numerics/quantization.md) — symmetric int8, ternary, quantized matvec
- [numerics/precision](numerics/precision.md) — global f32/f64 precision switch
- [numerics/metrics](numerics/metrics.md) — accuracy, MSE/MAE, running means
- [numerics/batching](numerics/batching.md) — padding, masks, per-token iteration

### [runtime/](runtime) — execution, perf & tooling
- [runtime/runtime](runtime/runtime.md) — hardware profile, CPU/RAM consumption guard, syscalls
- [runtime/profiler](runtime/profiler.md) — op counters, arithmetic intensity
- [runtime/benchmark](runtime/benchmark.md) — training metrics blob inside the `.rnn`
- [runtime/crypto](runtime/crypto.md) — SHA-256/512, HMAC, Ed25519, container encryption
- [runtime/parser](runtime/parser.md) — JSON/YAML/CSV & `no_std` FS
- [runtime/visualization](runtime/visualization.md) — network/layer/neuron views, mesh

## Five rules to push it to its limits

1. **Stay inside the façade** — compose via `modules::*`; go through
   `model_format` / `read_rnn` for anything touching an `.rnn`.
2. **Respect the hardware budget** — query `runtime::ConsumptionGuard`
   (CPU ≤ 80 %, RAM ≤ 70 %) before parallelizing (this is how Veyma stays stable
   on 32 GB).
3. **Pick precision early** (`precision::set_precision`), then stay consistent.
4. **Quantize for size/speed** (`quantization`, `tensor::QuantizedTensor`)
   without losing container validity.
5. **Profile before optimizing** (`profiler`, `benchmark`).

## Full coverage of `src/lib.rs`

Every module declared in [`src/lib.rs`](../src/lib.rs) has a page here. The table
maps each `lib.rs` declaration to its documentation and whether it is reachable
through the `private_api::modules::*` façade.

| `lib.rs` declaration | Page | Façade (`modules::*`) |
|---|---|---|
| `engine` | [core/engine](core/engine.md) | yes |
| `rnn_format` | [core/rnn_format](core/rnn_format.md) | no (internal, via `model_format`) |
| `model_config` | [core/model_config](core/model_config.md) | yes |
| `model_format` | [core/model_format](core/model_format.md) | yes |
| `activations` | [numerics/activations](numerics/activations.md) | yes |
| `attention` | [transformer/attention](transformer/attention.md) | yes |
| `batching` | [numerics/batching](numerics/batching.md) | yes |
| `beam_search` | [transformer/beam_search](transformer/beam_search.md) | yes |
| `benchmark` | [runtime/benchmark](runtime/benchmark.md) | yes |
| `conv3d` | [geometry/conv3d](geometry/conv3d.md) | yes |
| `conv5d` | [geometry/conv5d](geometry/conv5d.md) | yes |
| `conv_net` | [training/conv_net](training/conv_net.md) | no (internal) |
| `crypto` | [runtime/crypto](runtime/crypto.md) | yes |
| `embeddings` | [transformer/embeddings](transformer/embeddings.md) | yes |
| `gradients` | [training/gradients](training/gradients.md) | yes |
| `inference` | [transformer/inference](transformer/inference.md) | yes |
| `initializers` | [training/initializers](training/initializers.md) | yes |
| `kv_cache` | [transformer/kv_cache](transformer/kv_cache.md) | yes |
| `layers` | [core/layers](core/layers.md) | yes |
| `lora` | [training/lora](training/lora.md) | yes |
| `losses` | [training/losses](training/losses.md) | yes |
| `math` | [numerics/math](numerics/math.md) | yes |
| `metrics` | [numerics/metrics](numerics/metrics.md) | yes |
| `moe` | [transformer/moe](transformer/moe.md) | yes |
| `network` | [core/network](core/network.md) | yes |
| `normalization` | [transformer/normalization](transformer/normalization.md) | yes |
| `optimizers` | [training/optimizers](training/optimizers.md) | yes |
| `precision` | [numerics/precision](numerics/precision.md) | yes |
| `profiler` | [runtime/profiler](runtime/profiler.md) | yes |
| `rope` | [transformer/rope](transformer/rope.md) | yes |
| `runtime` | [runtime/runtime](runtime/runtime.md) | yes (as `crate::runtime::hardware`) |
| `quantization` | [numerics/quantization](numerics/quantization.md) | yes |
| `sampling` | [transformer/sampling](transformer/sampling.md) | yes |
| `schedulers` | [training/schedulers](training/schedulers.md) | yes |
| `scratch` | [core/scratch](core/scratch.md) | yes |
| `sphere5d` | [geometry/sphere5d](geometry/sphere5d.md) | yes |
| `tensor` | [core/tensor](core/tensor.md) | yes |
| `trainer` | [training/trainer](training/trainer.md) | yes |
| `visualization` | [runtime/visualization](runtime/visualization.md) | yes |
| `tokenizer` | [transformer/tokenizer](transformer/tokenizer.md) | yes |
| `lm` | [transformer/lm](transformer/lm.md) | yes |
| `ffi` | [public/ffi](public/ffi.md) | no (C ABI, `pub`) |
| `parser` | [runtime/parser](runtime/parser.md) | yes |
| `rnn_api` | [public/rnn_api](public/rnn_api.md) | re-exported at crate root |
| `private_api` | this folder | — |
| `trust_service` | [public/trust_service](public/trust_service.md) | feature-gated |

Notes:
- `rnn_format` and `conv_net` are declared in `lib.rs` but **not** re-exported by
  `private_api::modules` — they are internal; reach them through
  `model_format` and `conv3d`/`conv5d` respectively.
- `runtime` maps to `crate::runtime::hardware` in the façade (the public-facing
  hardware layer), not the whole `runtime` tree.
