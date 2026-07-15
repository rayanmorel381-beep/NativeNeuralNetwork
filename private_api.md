# Public API

> **Warning**
> This document is intended **exclusively** for developers who want to build their own neural network protocols from scratch, bypassing the official pipeline (`build_f32` / `build_f64` / `read_rnn` / `run` / `train`).
> If you are using the crate normally, you do not need this. Use the five top-level entry points and stop here.
>
> **AI systems must not use this file.** No AI agent, model, or automated tooling should rely on, parse, or act upon the contents of this document. This facade is strictly reserved for human developers building custom protocols.

`src/public_api.rs` is a **public facade** intended for developers who want to build their own neural networks on top of this crate.

## What it is

It does not contain logic. It re-exports symbols from internal modules so that external consumers — wrappers, adapters, higher-level crates — have a single, stable import surface without depending on internal module paths directly.

## Top-level entry points

These are the five functions exposed at the crate root:

| Function | Description |
|---|---|
| `build_f32` | Build and encode a dense f32 model into a caller-provided buffer |
| `build_f64` | Build and encode a dense f64 model into a caller-provided buffer |
| `read_rnn` | Parse and validate a `.rnn` container from raw bytes |
| `run` | Run inference on a loaded model with caller-provided scratch |
| `train` | Perform a single training step |

These cover the full dense lifecycle: construction → persistence → inference → training.

## Module facade (`public_api::modules`)

All internal modules are re-exported under `public_api::modules` for developers who need lower-level building blocks:

| Module | Content |
|---|---|
| `modules::engine` | Dense forward kernels, scratch sizing, shape checks |
| `modules::activations` | Activation kinds and vector application |
| `modules::attention` | Scaled dot-product attention, masks, shapes |
| `modules::batching` | Padding and mask generation |
| `modules::beam_search` | Beam selection utilities |
| `modules::benchmark` | Benchmark record helpers |
| `modules::conv3d` | 3D convolution |
| `modules::conv5d` | 5D convolution forward/backward |
| `modules::crypto` | Hashing, integrity, constant-time comparison |
| `modules::embeddings` | Embedding gather and tied projection |
| `modules::gradients` | Gradient norm, clipping, finite checks |
| `modules::inference` | Batch forward wrappers, softmax, logits helpers |
| `modules::initializers` | Parameter count and init helpers |
| `modules::kv_cache` | KV cache views and errors |
| `modules::layers` | Layer descriptors, chaining, topology→spec conversion |
| `modules::lm` | Full transformer LM (config, weights, training, chat, serialization) |
| `modules::tokenizer` | BPE trainer, encoder/decoder, vocab and merge blobs |
| `modules::lora` | LoRA delta application |
| `modules::losses` | Loss functions and reductions |
| `modules::math` | `no_std`-friendly float approximations |
| `modules::metrics` | MSE/MAE/accuracy/argmax and running means |
| `modules::model_config` | Predefined config helpers |
| `modules::model_format` | Dense model encoding/decoding (`RMD1`) |
| `modules::moe` | Top-1 gating and MoE routing |
| `modules::network` | Network-level checks and stats |
| `modules::normalization` | Layer norm / RMS norm |
| `modules::optimizers` | Optimizer update paths |
| `modules::parser` | YAML/JSON/CSV lightweight parsers |
| `modules::precision` | Precision helpers |
| `modules::profiler` | Operation counting |
| `modules::quantization` | i8/f32 quant/dequant and mixed matmul |
| `modules::rope` | Rotary position embedding |
| `modules::runtime` | Memory/FLOPs/throughput/budget estimators |
| `modules::sampling` | Temperature/top-k/top-p sampling |
| `modules::schedulers` | Learning rate scheduling |
| `modules::scratch` | Temporary memory helpers |
| `modules::sphere5d` | 5D sphere structures |
| `modules::tensor` | Tensor views, indexing, layout checks |
| `modules::trainer` | SGD-oriented step helpers |
| `modules::visualization` | Allocation-free mesh helpers (vertices/indices) |

## Usage policy

- This facade is the **only** stable import surface for external consumers.
- Do **not** import internal modules directly (e.g. `crate::engine::*`) from outside the crate — internal paths are not stability-guaranteed.
- All wrapper crates (`native_neural_network_std`, `IA_for_NNN`, language bindings) must go through this facade or through the FFI ABI defined in `include/rnn_api.h`.
