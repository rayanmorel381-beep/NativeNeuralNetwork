# Cookbook — combining modules end-to-end

The per-module pages show one capability each. Real work combines several. These
recipes chain modules the way Veyma does in production. Every recipe stays inside
the façade and never hand-crafts an `.rnn`.

Prereqs used below:

```rust
use native_neural_network::rnn_api::{build_f32, train, run, read_rnn};
use native_neural_network::private_api::modules as m;
```

---

## 1. Train from scratch, then read metrics back

Modules: `network` → `initializers` → **build** → **train** → `benchmark`.

```rust
const TOPOLOGY: &[usize] = &[128, 256, 128, 64];
const LM_PARAMS: &[usize] = &[32768, 1024, 2048, 4096, 16, 16, 8, 128, 16384, 512];

let n_w = m::network::expected_weights_count(TOPOLOGY);
let n_b = m::network::expected_biases_count(TOPOLOGY);
let mut weights = vec![0.0f32; n_w];
let mut biases = vec![0.0f32; n_b];
m::initializers::initialize_parameters_f32_he_uniform(TOPOLOGY, &mut weights, 0xA1B2_C3D4);

let mut model_buf = vec![0u8; 2 << 30];
let mut out = vec![0u8; 2 << 30];

let built = build_f32("m.rnn", TOPOLOGY, 1u64, &mut weights, &mut biases,
                      LM_PARAMS, &["dataset/en", "dataset/fr"], &mut model_buf).unwrap();
let trained = train(&mut model_buf[..built], 1800, &mut out).unwrap();

let view = read_rnn(&mut out[..trained], None).unwrap();
// view.avg_loss, view.iterations_per_sec, view.eval_accuracy, ...
```

---

## 2. Parallelize under the hardware budget

Modules: `runtime` (guard) → `trainer` (fork pool). This is *the* pattern that
keeps a 32 GB box responsive.

```rust
let guard = m::runtime::ConsumptionGuard::detect();

let slot = 32 * 1024 * 1024;                 // per-worker arena
let budget = guard.ram_budget_bytes();       // 70 % RAM cap
let mut workers = requested;
if budget > 0 { workers = workers.min((budget / slot).max(1)); }
workers = guard.clamp_workers(workers);      // 80 % CPU cap

// hand `workers` to the trainer pool — idle workers block on a futex (0 % CPU).
```

---

## 3. Chat generation with a KV-cache

Modules: `tokenizer` → `lm` → `inference` → `attention`/`rope`/`kv_cache` →
`sampling`.

```rust
// 1) encode the prompt
let mut ids = [0u32; 512];
let n = m::tokenizer::encode_with_bpe(&vocab, &merges, prompt.as_bytes(), &mut ids).unwrap();

// 2) prefill once, then decode token-by-token reusing scratch + kv-cache
//    (inference::prefill_prompt + per-step decode)

// 3) pick each token
fn next_token(logits: &mut [f32]) -> usize {
    m::sampling::softmax_temperature(logits, 0.8);
    m::sampling::top_k_mask(logits, 40);
    let cut = m::sampling::top_p_cutoff(logits, 0.95);
    m::sampling::sample_from_cumulative(logits, cut, draw_u01())
}

// 4) decode ids back to text
let mut text = [0u8; 4096];
let _ = m::tokenizer::decode_ids_utf8(&vocab, &ids[..n], &mut text);
```

For deterministic output, replace steps in `next_token` with
`m::sampling::argmax_sample(logits)`.

---

## 4. Quantize a trained model for size/speed

Modules: `quantization` → `tensor` → `model_format` (stays a valid `.rnn`).

```rust
// int8-symmetric a weight matrix, then run matvecs on the packed form
let mut qw = vec![0i8; weights.len()];
let scale = m::quantization::quantize_i8_symmetric(&weights, &mut qw);   // ~4x smaller

fn linear_i8(qw: &[i8], scale: f32, x: &[f32], y: &mut [f32]) {
    m::quantization::matvec_w8(qw, scale, x, y);   // no dequant in the hot path
}
```

Ternary packs even tighter for tolerant layers (`quantize_ternary` +
`matvec_ternary`).

---

## 5. LoRA adaptation on a frozen base

Modules: `lora` → `inference` (`generate_lora`). Rank `r` is the only cost knob.

```rust
fn adapted_linear(a: &[f32], b: &[f32], x: &[f32], y: &mut [f32], scale: f32) {
    let mut delta = vec![0.0f32; y.len()];
    m::lora::lora_forward_delta(a, b, x, &mut delta);   // B·A·x
    m::lora::apply_lora_delta(y, &delta, scale);        // y += scale · delta
}
```

Keep the base weights quantized (recipe 4) and only the LoRA factors in f32.

---

## 6. Roofline: is my forward pass compute- or memory-bound?

Modules: `profiler` → decide, then optimize the right axis.

```rust
let mut ops = m::profiler::OpCounter::default();
ops.add_matmul(m_, n_, k_);
ops.add_memory_read((m_ * k_ + k_ * n_) * 4);
ops.add_memory_write(m_ * n_ * 4);

let intensity = ops.arithmetic_intensity();          // FLOP/byte
let bound = if ops.is_memory_heavy() { "memory" } else { "compute" };
```

---

## 7. Inspect / visualize a trained network

Modules: `visualization` (zero-copy views) → `sphere5d` (5D geometry).

```rust
let net = m::visualization::get_network_view(rnn_bytes).unwrap();
for l in 0..net.layer_count() {
    let layer = m::visualization::layer(&net, l);
    let _ = (layer.neuron_count(), layer.weight_count(), layer.activation());
}

let (v, i) = m::visualization::mesh_required_buffers_from_bytes(rnn_bytes);
let mut vbuf = vec![0.0f32; v];
let mut ibuf = vec![0u32; i];
m::visualization::fill_mesh_from_bytes(rnn_bytes, &mut vbuf, &mut ibuf);
```

---

## 8. Verify integrity before running an untrusted `.rnn`

Modules: `model_format` (header check) → `crypto` (signature, owner-bound).

```rust
// structural check first
let header = m::model_format::parse_header(bytes).unwrap();
assert!(m::model_format::is_header_consistent(&header));

// then authenticity (constant-time)
assert!(m::crypto::ed25519_verify(public_key, message, signature));
// device-bound models: decrypt_rnn_payload_owner_bound(bytes, key, device_id)
```

Never compare digests/tags with `==` — use `m::crypto::constant_time_eq`.

---

## 9. The grand tour — one AI that touches every module

A single narrative: **"Aria"**, an assistant that is built, secured, trained,
inspected, quantized, adapted and served. Each numbered stage names the modules
it exercises, so by the end every module declared in `lib.rs` has been used at
least once. The code is illustrative pseudo-Rust — real signatures live on each
module page — but the call flow is exactly what Veyma runs.

```rust
use native_neural_network::rnn_api::{
    build_f32, build_f64, train, run, read_rnn, BuildF32Request,
};
use native_neural_network::private_api::modules as m;
// feature-gated: use native_neural_network::trust_service as trust;
```

### Stage 0 — Read the environment  (`runtime`, `precision`, `math`)

```rust
// Detect hardware and lock in the compute budget (CPU ≤ 80 %, RAM ≤ 70 %).
let guard = m::runtime::ConsumptionGuard::detect();
let ram_budget = guard.ram_budget_bytes();

// Pick precision once for the whole life of the model.
m::precision::set_precision(m::precision::Precision::F32);

// `math` underpins everything numeric below (no_std sin/exp/tanh/…).
let _warm = m::math::tanhf(0.5);
```

### Stage 1 — Design the brain  (`layers`, `network`, `initializers`, `scratch`, `tensor`)

```rust
const TOPOLOGY: &[usize] = &[128, 256, 128, 64];
assert!(m::layers::layer_chain_is_compatible(TOPOLOGY));

let n_w = m::network::expected_weights_count(TOPOLOGY);
let n_b = m::network::expected_biases_count(TOPOLOGY);
let mut weights = vec![0.0f32; n_w];
let mut biases  = vec![0.0f32; n_b];

// Reproducible He init.
m::initializers::initialize_parameters_f32_he_uniform(TOPOLOGY, &mut weights, 0xA11A);

// One aligned arena, reused for every forward/backward step (no per-step alloc).
let widest = m::layers::max_width(TOPOLOGY);
let mut backing = vec![0.0f32; widest * 8];
let mut arena = m::scratch::Scratch::new(bytemuck_cast(&mut backing));

// Wrap parameters as tensors for the compute layer.
let w_view = m::tensor::DenseTensor::from_slice(&weights);
let _ = (w_view, &arena);
```

### Stage 2 — Language brain & tokenizer  (`lm`, `tokenizer`, `embeddings`)

```rust
const LM_PARAMS: &[usize] = &[32768, 1024, 2048, 4096, 16, 16, 8, 128, 16384, 512];

// Encode a seed prompt (BPE over the trained vocab/merges).
let mut ids = [0u32; 512];
let n_ids = m::tokenizer::encode_with_bpe(&vocab, &merges, b"Hello", &mut ids).unwrap_or(0);

// Embedding lookup turns ids into vectors at the input.
let mut hidden = vec![0.0f32; n_ids * 1024];
m::embeddings::gather_embeddings(&embed_table, &ids[..n_ids], 1024, &mut hidden);
```

### Stage 3 — Build & secure the container  (`build`, `model_format`, `rnn_format`, `crypto`, `benchmark`)

```rust
let mut model_buf = vec![0u8; 2 << 30];
let built = build_f32(
    "aria.rnn", TOPOLOGY, 1u64, &mut weights, &mut biases,
    LM_PARAMS, &["dataset/en", "dataset/fr"], &mut model_buf,
).unwrap();

// The container is now a valid RMD1: validate its structure & signature.
let header = m::model_format::parse_header(&model_buf[..built]).unwrap();
assert!(m::model_format::is_header_consistent(&header));   // model_format
assert!(m::rnn_format::validate(&model_buf[..built], None).is_ok()); // rnn_format
// benchmark blob is embedded by the build; crypto guards integrity:
// m::crypto::ed25519_verify(pk, msg, sig)
```

### Stage 4 — Train under budget  (`trainer`, `optimizers`, `losses`, `gradients`, `schedulers`, `batching`, `metrics`)

```rust
// One call runs the whole loop: it pads batches (batching), computes
// cross-entropy (losses), clips grads (gradients), steps AdamW (optimizers)
// on a warmup+cosine schedule (schedulers), forks workers clamped by the
// runtime guard, and accumulates eval metrics (metrics) into the benchmark blob.
let mut out = vec![0u8; 2 << 30];
let workers = guard.clamp_workers(requested_workers);      // runtime + trainer
let trained = train(&mut model_buf[..built], 1800, &mut out).unwrap();
let _ = workers;
```

### Stage 5 — Introspect the trained mind  (`visualization`, `sphere5d`, `profiler`)

```rust
let view = read_rnn(&mut out[..trained], None).unwrap();     // benchmark metrics
let net_view = m::visualization::get_network_view(&out[..trained]).unwrap();
let layers = net_view.layer_count();

// Project neurons into 5D and measure their spread.
// let sphere = m::sphere5d::Sphere5D::from_network(&network);
// let spread = sphere.mean_radius();

// Roofline: is the forward pass compute- or memory-bound?
let mut ops = m::profiler::OpCounter::default();
ops.add_matmul(256, 128, 128);
let _bound = ops.is_memory_heavy();
let _ = (view, layers);
```

### Stage 6 — Shrink & specialize  (`quantization`, `lora`, `conv_net`, `conv3d`, `conv5d`)

```rust
// Quantize weights to int8 for a ~4× smaller deployable model.
let mut qw = vec![0i8; weights.len()];
let scale = m::quantization::quantize_i8_symmetric(&weights, &mut qw);

// Adapt the frozen base to a new domain with a cheap low-rank delta.
let mut delta = vec![0.0f32; 64];
m::lora::lora_forward_delta(&lora_a, &lora_b, &hidden[..], &mut delta);

// A vision side-branch runs convolutions (conv_net config → conv3d/conv5d).
// let shape = m::conv3d::conv3d_output_shape(&conv_args);
let _ = (scale, delta);
```

### Stage 7 — Think: one generation step  (`inference`, `attention`, `rope`, `kv_cache`, `normalization`, `moe`, `activations`, `beam_search`, `sampling`)

```rust
// Per token: normalize (normalization) → rotate Q/K (rope) → attend over the
// cache (attention + kv_cache) → route experts (moe) → activate (activations)
// → project to logits → decode (sampling, or beam_search for quality).
fn think_step(logits: &mut [f32], cache_full: bool) -> Option<usize> {
    if cache_full { return None; }                       // kv_cache guard
    m::normalization::rms_norm_in_place(logits, &gamma, 1e-5);   // normalization
    m::activations::apply_in_place(m::activations::ActivationKind::from_u8(3), logits);
    m::sampling::softmax_temperature(logits, 0.8);       // sampling
    m::sampling::top_k_mask(logits, 40);
    let cut = m::sampling::top_p_cutoff(logits, 0.95);
    Some(m::sampling::sample_from_cumulative(logits, cut, draw_u01()))
}
// Deterministic path: m::sampling::argmax_sample(logits)
// Quality path:       m::inference::generate_beam(...) + m::beam_search::select_top_beams(...)
```

### Stage 8 — Answer & decode  (`tokenizer`, `run`)

```rust
// The public `run` drives the full prefill→decode flow internally.
let mut cfg = /* ModelConfig */ default_config();
let (score, out_len) = run(&mut out[..trained], &mut cfg).unwrap();

// Turn generated ids back into UTF-8 text.
let mut text = [0u8; 4096];
let _m = m::tokenizer::decode_ids_utf8(&vocab, &ids[..n_ids], &mut text);
let _ = (score, out_len);
```

### Stage 9 — Ship it  (`ffi`, `parser`, `trust_service`, `model_config`)

```rust
// Load runtime settings from a JSON config (bounded, no_std).
let _settings = m::parser::parse_json_with_max_depth(config_json, 16);

// Cross-language serving goes through the C ABI (ffi::api's 5 symbols),
// which the Python/JS/Java/C++ wrappers call.
//   rnn_ffi_run(...), rnn_ffi_read_rnn(...) -> RnnFfiReadView

// With the publisher-trust-service feature, sign the distribution:
// #[cfg(feature = "publisher-trust-service")]
// trust::PublisherTrustService::sign_model_distribution(...);
```

### Module checklist

Stages above collectively touch **every** `lib.rs` module:

`runtime` `precision` `math` · `layers` `network` `initializers` `scratch`
`tensor` · `lm` `tokenizer` `embeddings` · `model_format` `rnn_format` `crypto`
`benchmark` · `trainer` `optimizers` `losses` `gradients` `schedulers` `batching`
`metrics` · `visualization` `sphere5d` `profiler` · `quantization` `lora`
`conv_net` `conv3d` `conv5d` · `inference` `attention` `rope` `kv_cache`
`normalization` `moe` `activations` `beam_search` `sampling` · `engine`
(under every compute call) · `model_config` `parser` `ffi` `rnn_api`
`trust_service`.

> This is a map, not a compilable program: the placeholders (`vocab`, `merges`,
> `embed_table`, `gamma`, `default_config`, …) stand in for your real state. Each
> individual call matches the signatures on the corresponding module page.

