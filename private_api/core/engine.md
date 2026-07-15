# core/engine

Backend dispatch and kernel contracts. `engine` decides **where** a compute op
runs (CPU, GPU, TPU, LPU) and enforces a capability contract before dispatching.

Façade: `private_api::modules::engine`.

## Key types

- `ComputeBackend` — the available compute targets.
- `BackendContractCapabilities` — what a backend guarantees (features/flags).
- `ContractRejectReason` / `ContractRejectHandler` — how an op is refused when a
  backend cannot honor the contract.
- `*CmdF32` / `*CmdF64` / `*Invoke*` families (`AdamwCmd*`, `AttentionCmd*`,
  `DequantizeI8Cmd*`, …) — typed command + invocation descriptors per op.
- `BackendTimingAbstraction`, `BufferStagingProfile` — timing and staging models.
- `EvolveConfig` — progressive backend discovery configuration.

## Contract flags & features

- Features: `CONTRACT_FEATURE_F32`, `CONTRACT_FEATURE_F64`,
  `CONTRACT_FEATURE_INT8`, `CONTRACT_FEATURE_BATCH`.
- Flags: `CONTRACT_FLAG_DETERMINISTIC_MATH`, `CONTRACT_FLAG_REQUIRE_FINITE_INPUTS`,
  `CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT`, mask `CONTRACT_KNOWN_FLAGS_MASK`.
- Discovery stages: `DISCOVERY_STAGE_*`.
- Hardware defaults: `DEFAULT_GPU_*`, `DEFAULT_TPU_*`, `DEFAULT_LPU_*`,
  `DEFAULT_HARDWARE_HEADROOM_PPM`, `FLOPS_SCALE_GIGA`.

## Concrete usage

Build a strict contract before dispatching an int8 batched matmul, so a backend
that cannot guarantee finite inputs / alignment is rejected instead of silently
producing garbage:

```rust
use native_neural_network::private_api::modules::engine::{
    CONTRACT_FEATURE_INT8, CONTRACT_FEATURE_BATCH,
    CONTRACT_FLAG_REQUIRE_FINITE_INPUTS, CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT,
};

let required_features = CONTRACT_FEATURE_INT8 | CONTRACT_FEATURE_BATCH;
let required_flags =
    CONTRACT_FLAG_REQUIRE_FINITE_INPUTS | CONTRACT_FLAG_REQUIRE_STRICT_ALIGNMENT;

// A backend is only eligible if it advertises every required bit.
fn backend_is_eligible(caps_features: u32, caps_flags: u32) -> bool {
    caps_features & required_features == required_features
        && caps_flags & required_flags == required_flags
}
```

## Performance levers

- GPU is only chosen when it actually helps (batched paths, `batch_size > 1`);
  otherwise the CPU kernel is preferred. Do not force GPU for tiny work — that is
  what makes Veyma's dispatch stable.
- Require `CONTRACT_FLAG_REQUIRE_FINITE_INPUTS` for untrusted data.
- Use `CONTRACT_FLAG_DETERMINISTIC_MATH` when you need reproducible results
  across backends (at some throughput cost).

## Integration

`engine` is what [tensor](../core/tensor.md), [attention](../transformer/attention.md)
and [trainer](../training/trainer.md) route through. It reads hardware limits
from [runtime](../runtime/runtime.md).
