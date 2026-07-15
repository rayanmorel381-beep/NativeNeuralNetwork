# core/scratch

A pre-allocated, aligned memory arena. In a `#![no_std]` crate with no dynamic
allocation on the hot path, `scratch` is how you get scratch buffers without
touching a global allocator.

Façade: `private_api::modules::scratch`.

## Key type

- `Scratch` — an arena you carve slices out of.

## Functions

- `new(...)` — wrap a backing buffer as an arena.
- `alloc_align(...)` — carve an aligned sub-slice.
- `as_slice()` — view the arena.
- `base_ptr()` — raw base pointer for interop.

## Concrete usage

Allocate one big arena, then carve aligned working buffers for successive stages
without ever hitting the allocator again:

```rust
use native_neural_network::private_api::modules::scratch::Scratch;

let mut backing = vec![0u8; 1 << 20];       // 1 MiB, allocated once
let mut arena = Scratch::new(&mut backing);

// 64-byte aligned region for a SIMD/GPU-friendly kernel.
let a = arena.alloc_align(4096, 64);
// carve more regions from the same arena as the pipeline advances...
let _ = a;
```

## Performance levers

- Size one `Scratch` from [layers](../core/layers.md) `max_width` /
  [inference](../transformer/inference.md) `*_f32_count` helpers, then reuse it —
  zero per-iteration allocation.
- Use `alloc_align` to satisfy the strict-alignment contract in
  [engine](../core/engine.md).

## Integration

Backs the forward/backward scratch in [inference](../transformer/inference.md)
and [trainer](../training/trainer.md).
