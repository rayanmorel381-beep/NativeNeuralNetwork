# nnn-base

`nnn-base` is the fundamental foundation of `native_neural_network`.

It provides the low-level primitives required by the entire ecosystem, including memory management, mathematical operations, activation functions, initialization mechanisms, and tensor operations.

Its architecture is built around three fundamental principles:

1. **`scratch` is the foundation of the entire architecture.** It completely replaces `alloc` as the memory management infrastructure used internally by `native_neural_network`.
2. **All other functional modules rely on `scratch`.** No module may introduce an independent memory management infrastructure.
3. **`base.rs` is the sole public facade of `nnn-base`.** It constructs `LmlpBase`, which serves as the foundation for the rest of `native_neural_network`.

**`nnn-base` has no external dependencies and must never acquire any.**

---

## 1. Architectural Foundation

`nnn-base` occupies the lowest architectural level of `native_neural_network`.

It must remain completely independent of other crates, external frameworks, third-party mathematical libraries, tensor libraries, and external allocation systems.

Higher-level components may consume the capabilities exposed by `nnn-base`, but `nnn-base` itself must remain self-contained.

### Core Principles

- Complete independence from external crates.
- No external dependencies, direct or transitive.
- Centralized memory management through `scratch`.
- Internal functional modules built upon `scratch`.
- Centralized construction of `LmlpBase` in `base.rs`.
- A single public facade for the entire crate.
- Strict encapsulation of internal implementation details.
- No duplication of fundamental mechanisms across modules.

---

## 2. `scratch`: The Foundation of `native_neural_network`

The `scratch` module is the cornerstone of the memory architecture of `native_neural_network`.

It completely replaces `alloc` as the project's internal memory management infrastructure.

Its purpose is to centralize the primitives required to organize, manage, allocate, reuse, and access memory within an infrastructure controlled by `native_neural_network`.

Other modules must not independently reimplement these mechanisms or introduce competing memory management systems.

### Module Structure

```text
scratch/
├── align.rs
├── arena.rs
├── carve.rs
├── descriptor.rs
├── frame.rs
├── galloc.rs
├── layout.rs
├── logbuf.rs
├── mod.rs
├── mut_ptr.rs
├── pool.rs
├── slot.rs
└── typed.rs
```

Each component contributes to the internal organization of the memory subsystem.

| Component | Architectural Responsibility |
|---|---|
| `align.rs` | Memory alignment primitives. |
| `arena.rs` | Arena-based memory organization. |
| `carve.rs` | Memory region subdivision. |
| `descriptor.rs` | Memory and resource descriptors. |
| `frame.rs` | Memory frame organization. |
| `galloc.rs` | Internal allocation mechanisms managed by `scratch`. |
| `layout.rs` | Memory layout representation and organization. |
| `logbuf.rs` | Internal buffer management. |
| `mut_ptr.rs` | Mutable pointer handling. |
| `pool.rs` | Resource pooling and reuse. |
| `slot.rs` | Memory slot organization. |
| `typed.rs` | Typed memory resource management. |

These descriptions establish the intended architectural responsibilities of the components. Their exact behavior is determined by their implementations.

### Fundamental Rule

**Every internal memory management requirement within `nnn-base` must be handled through the `scratch` infrastructure.**

No functional module may bypass this architecture by introducing an independent allocation system.

`scratch` must not depend on `math`, `tensor`, `activations`, `initializers`, or `LmlpBase`.

It is the common foundation upon which these components are built.

---

## 3. `math`: Mathematical Primitives Built on `scratch`

The `math` module provides the mathematical primitives required by `native_neural_network`.

It relies on `scratch` whenever its operations require internal memory management.

It must not introduce its own competing memory infrastructure or depend on external mathematical libraries.

### Module Structure

```text
math/
├── elementwise.rs
├── exp_log.rs
├── float.rs
├── helpers.rs
├── logistic.rs
├── mod.rs
├── power.rs
├── roots.rs
├── rounding.rs
└── trig.rs
```

The module covers:

- Element-wise operations.
- Exponential and logarithmic functions.
- Floating-point operations.
- Mathematical helper functions.
- Logistic functions.
- Power operations.
- Root functions.
- Rounding operations.
- Trigonometric functions.

All mathematical primitives must remain implemented and controlled within `nnn-base`.

### Dependency Direction

```text
scratch
   ↑
 math
```

`math` may use the capabilities provided by `scratch`.

`scratch` must never depend on `math`.

This dependency direction is an architectural invariant and must remain unchanged as the mathematical subsystem evolves.

---

## 4. All Functional Modules Depend on `scratch`

The same architectural principle applies to every functional module within `nnn-base`.

No module may introduce an independent memory management infrastructure.

### `activations`

```text
activations/
├── functions.rs
├── mod.rs
└── numeric_guard.rs
```

This module contains activation functions and associated numerical safeguards.

It relies on `scratch` whenever its operations require internal memory management.

### `initializers`

```text
initializers/
├── init_core.rs
├── init_guard.rs
└── mod.rs
```

This module provides initialization mechanisms for the structures and parameters used by the neural network.

All internal memory requirements must be handled through `scratch`.

### `tensor`

```text
tensor/
├── cache_tensor.rs
├── decompose.rs
├── dense.rs
├── elementwise.rs
├── gpu.rs
├── lazy.rs
├── linalg.rs
├── mod.rs
├── packed.rs
├── quantized.rs
├── readonly.rs
├── shared.rs
├── sparse.rs
├── static_tensor.rs
├── tensor_core.rs
├── training.rs
└── view_mut.rs
```

The `tensor` module provides the primitives required for tensor representation and manipulation.

Its responsibilities include:

- Dense and sparse tensor representations.
- Element-wise tensor operations.
- Linear algebra operations.
- Packed and quantized representations.
- Lazy evaluation.
- Read-only and mutable views.
- Shared tensor representations.
- Tensor caching.
- Training-related tensor operations.
- GPU-related tensor functionality.

All internal memory management must respect the `scratch` architecture.

The presence of `gpu.rs` does not authorize introducing an external GPU dependency into `nnn-base`.

Any capability that genuinely requires an external library must be isolated in an appropriate crate outside `nnn-base`.

### Shared Architectural Principle

```text
                    scratch
                       |
          +------------+------------+
          |            |            |
        math       activations  initializers
          |            |            |
          +------------+------------+
                       |
                     tensor
```

This diagram illustrates the shared architectural foundation. It does not imply that the functional modules must depend on one another.

Each module must use `scratch` directly for the memory primitives it requires.

No functional module may become the owner of an independent memory management system.

The same rule applies to every future module added to `nnn-base`.

---

## 5. `base.rs`: The Sole Public Facade

`base.rs` has a unique architectural responsibility.

**It is the sole public facade of `nnn-base` and the designated location for constructing `LmlpBase`.**

`LmlpBase` is the foundational structure provided to the rest of `native_neural_network`.

Consumers must not need to understand the internal organization of `nnn-base` to construct or use this foundation.

### Responsibilities of `base.rs`

- Define and construct `LmlpBase`.
- Centralize the public entry point of the crate.
- Coordinate access to the internal capabilities required by `LmlpBase`.
- Preserve the encapsulation of internal modules.
- Provide a coherent public interface to the rest of `native_neural_network`.
- Prevent consumers from depending directly on internal implementation details.

### Internal Module Encapsulation

The following modules are internal implementation components:

```text
scratch
math
activations
initializers
tensor
```

Their internal implementation details must not automatically become accessible to consumers of `nnn-base`.

The public facade must expose only the types and operations explicitly required by the rest of `native_neural_network`.

Internal modules must remain free to evolve without forcing changes to the public interface, provided that the `LmlpBase` contract remains satisfied.

### The Role of `lib.rs`

`lib.rs` is Rust's crate entry point and is responsible for declaring and re-exporting the public interface.

It must not become a second, independent public facade.

The public API must remain architecturally centralized around `base.rs`.

Rust's requirement for a crate root does not change this architectural rule.

---

## 6. `LmlpBase`: The Shared Foundation

`LmlpBase` is constructed in `base.rs` to serve as the foundation for the rest of `native_neural_network`.

Its construction must respect the following requirements:

1. It uses the internal capabilities provided by `nnn-base`.
2. It relies on the memory infrastructure provided by `scratch`.
3. It introduces no external dependency.
4. It does not unnecessarily expose internal implementation details.
5. It provides a coherent public contract to consuming crates.

The construction of `LmlpBase` must remain centralized.

Other modules and crates must not independently recreate this foundation or bypass the public facade to access its internal mechanisms.

`LmlpBase` is the architectural boundary between the internal primitives of `nnn-base` and the higher-level components of `native_neural_network`.

---

## 7. Absolute Prohibition of External Dependencies

**`nnn-base` must never have external dependencies.**

This is a permanent architectural constraint, not a temporary design preference.

No feature justifies adding an external dependency to this crate.

This prohibition includes dependencies for:

- Memory allocation.
- Mathematical operations.
- Tensor manipulation.
- Neural network frameworks.
- GPU functionality.
- General-purpose utilities.
- Any transitive dependency introduced by another crate.

Fundamental capabilities must be implemented within `nnn-base`, respecting the separation of its internal modules.

If a capability genuinely requires an external library, it must be isolated in another crate within the `native_neural_network` ecosystem.

That external dependency must never enter the dependency graph of `nnn-base`.

### Why This Constraint Matters

Keeping `nnn-base` dependency-free preserves:

- Control over the fundamental architecture.
- Independence from third-party implementations.
- Ownership of internal mechanisms.
- A consistent technical foundation.
- A clear separation between fundamental primitives and higher-level functionality.

This requirement does not prohibit the use of Rust's standard library or language-provided facilities.

It prohibits external crate dependencies, whether direct or transitive.

---

## 8. Architectural Hierarchy

The architecture of `nnn-base` follows a strict hierarchy.

```text
                 NATIVE_NEURAL_NETWORK
                          |
                          v
                       nnn-base
                          |
                          v
                        base.rs
                          |
                          v
                       LmlpBase
                          |
             +------------+------------+
             |                         |
             v                         v
      Internal Modules           Public Interface
             |
             v
           scratch
             |
       +-----+----------+
       |     |          |
       v     v          v
      math  tensor  activations
                         |
                         v
                    initializers
```

This diagram represents architectural responsibilities rather than a complete graph of direct dependencies.

The governing rules are:

- `scratch` provides the fundamental memory primitives.
- Functional modules rely on `scratch`.
- `base.rs` constructs and exposes `LmlpBase`.
- `lib.rs` makes the public facade available as the crate entry point.
- The rest of `native_neural_network` consumes the public interface of `nnn-base`.
- No external dependency is introduced into `nnn-base`.

---

## 9. Internal Tests

The `tests` directory contains tests for the fundamental capabilities of `nnn-base`.

```text
tests/
├── activations.rs
├── frame.rs
├── math.rs
├── mod.rs
├── pool.rs
├── scratch.rs
├── support.rs
├── tensor.rs
└── typed.rs
```

These tests help verify the behavior of fundamental primitives and preserve architectural consistency.

They should cover:

- `scratch` memory management mechanisms.
- Mathematical operations.
- Activation functions.
- Tensor manipulation.
- Typed memory management.
- The behavioral contracts of internal components.

Tests must not introduce external dependencies into `nnn-base`.

---

## 10. Rules for Future Development

Every modification to `nnn-base` must comply with the following rules.

### Rule 1 — `scratch` Is the Memory Foundation

Every new feature requiring internal memory management must integrate with `scratch`.

No competing memory infrastructure may be introduced into other modules.

### Rule 2 — Functional Modules Rely on `scratch`

`math`, `activations`, `initializers`, `tensor`, and every future functional module must respect this architecture.

No functional module may impose an independent memory management system on the rest of the crate.

### Rule 3 — `base.rs` Is the Sole Public Facade

Internal functionality must not be exposed directly to the rest of the ecosystem.

Any capability intended for consumers must be incorporated into the public interface through `base.rs` when required.

### Rule 4 — `LmlpBase` Is Constructed in `base.rs`

Construction of the public foundation must not be scattered across multiple modules or crates.

### Rule 5 — No External Dependencies

No external dependency may be added to `nnn-base`, directly or transitively.

### Rule 6 — No Duplication of Fundamental Mechanisms

A primitive already provided by `scratch` must not be independently reimplemented in another module.

### Rule 7 — Preserve Encapsulation

The internal file structure is not the public contract of `nnn-base`.

Consumers must use the public facade rather than depend on internal implementation details.

### Rule 8 — Preserve Dependency Direction

`scratch` must remain independent of all functional modules.

No reverse dependency may be introduced.

---

## Conclusion

`nnn-base` is the autonomous foundation of `native_neural_network`.

Its architecture separates fundamental memory management, functional primitives, and the public interface.

`scratch` is the cornerstone of this architecture and completely replaces `alloc` as the internal memory management infrastructure.

`math`, `activations`, `initializers`, `tensor`, and all other functional modules rely on this foundation.

`base.rs` is the sole public facade and centralizes the construction of `LmlpBase`, which serves as the foundation for the rest of `native_neural_network`.

Finally, **`nnn-base` must never have external dependencies, whether direct or transitive.**

This constraint ensures that the fundamental core of `native_neural_network` remains autonomous, encapsulated, and entirely under the control of its own ecosystem.