# public/ffi

The C ABI. `ffi` exposes the same five operations as
[rnn_api](../public/rnn_api.md), but as `extern "C"` symbols for the language
wrappers (C/C++, Python, JS, Java). It is a **thin dispatch layer** — no model
logic lives here.

Path: `native_neural_network::ffi::api`. Declared C prototypes:
[`include/rnn_api.h`](../../include/rnn_api.h).

## Exported symbols

Exactly five `#[no_mangle] pub extern "C"` functions plus one struct:

- `rnn_ffi_build_f32(...)` / `rnn_ffi_build_f64(...)` — build into a caller buffer.
- `rnn_ffi_train(...)` — train for N seconds.
- `rnn_ffi_run(...)` — inference.
- `rnn_ffi_read_rnn(...) -> RnnFfiReadView` — parse & validate.
- `RnnFfiReadView` — a C-friendly view over a parsed container (pointers + lens,
  benchmark scalars).

## Error codes

Returned as `int` status codes (see the header): `RNN_FFI_OK = 0`, with negative
codes for build/train/run/read failures.

## Concrete usage (C)

```c
#include "rnn_api.h"

int rc = rnn_ffi_build_f32(
    "m.rnn", topology, topology_len, seed,
    weights, weights_len, biases, biases_len,
    lm_params, lm_params_len, dataset_dirs, dataset_dirs_len,
    out_buf, out_cap, &out_len);
if (rc != RNN_FFI_OK) { /* handle */ }
```

The higher-level wrappers (`wrappers/{cpp,python,javascript,java}/rnn_api.*`) wrap
these five symbols; do not add new `extern "C"` functions here.

## Contract

- `ffi/` contains only `api.rs` (the five entry points) and an internal
  `dispatch.rs`. Keep it that way.
- Every call maps 1:1 to an [rnn_api](../public/rnn_api.md) function.

## Integration

The C boundary over [rnn_api](../public/rnn_api.md); consumed by the language
wrappers documented in [`wrappers/`](../../wrappers).
