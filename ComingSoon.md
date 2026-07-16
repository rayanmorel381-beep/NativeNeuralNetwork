# Coming Soon

I am actively preparing two major evolutions for this project: a unified neural graph protocol (**LMLP**) and an onboarding upgrade that removes setup friction.

The goal is simple: make the core more expressive under the hood, while helping everyone start quickly without building everything from scratch.

## What Is Being Prepared

### 1. LMLP — One Protocol for Every Model

Today the `.rnn` format effectively carries two shapes — a dense model (MLP) and a language model (LM) — each discovered by inspecting which blobs are present. **LMLP** (*Language Multi Layer Perceptron*) replaces that split with a single execution model and a single serialization format.

With LMLP, dense, sequential, wave, and event behaviors are no longer separate model classes. They **emerge** from three small primitives:

- **Signal** — the unit that flows. It carries its payload, its logical time, its confidence, and — crucially — its own *propagation mode* (`Dense`, `Sequential`, `Wave`, or `Event`).
- **Node** — a pure transformer `Signal → Signal`. It applies its parameters, its non-linearity, and its internal memory, and never knows whether it is serving an MLP or an LM.
- **Edge** — a router. It reads the mode carried by the signal and applies the matching transport law (delay, phase, decay).

The words "MLP" and "LM" disappear from the graph's vocabulary. What used to be different architectures become different *configurations* of the same graph.

Planned benefits:
- one execution path instead of per-model-type branching,
- dense, recurrent, wave, and event dynamics under a single protocol,
- a single graph descriptor inside the `.rnn` container (`lmlp.graph` for topology, `lmlp.tensors` for weights),
- new modes added by extending one table — not by rewriting the engine.

### 2. Backward Compatibility, Guaranteed

Existing `.rnn` files — dense and LM alike — stay fully readable. A compatibility adapter reconstructs the same in-memory graph from legacy blobs, and the unified forward pass is validated to be **bit-for-bit identical** to today's output on every existing sample and trained model.

During the transition, newly written models can carry both the new `lmlp.*` description and the legacy blobs, so tools that have not yet migrated keep working.

### 3. Hosted Server Environment

I am working on a dedicated server setup to make integration and first runs easier.

Planned benefits:
- faster project onboarding,
- simpler environment setup,
- standardized runtime behavior,
- easier testing and demonstration flows.

### 4. Pre-trained Model Pack(s)

I am preparing one or more pre-trained models that can be used immediately.

Planned benefits:
- immediate experimentation,
- no need to train from zero for first use,
- baseline references for evaluation,
- easier cross-language wrapper validation.

## Why This Matters

Today the core carries dense and language models as two distinct on-disk shapes, and new users often spend too much time on setup and initial training before they can test real workflows.

This upcoming update is intended to remove both frictions by providing:
- one protocol and one container for every model shape,
- ready-to-use infrastructure,
- ready-to-use model artifacts,
- clearer first steps for production-style workflows.

## Planned Rollout

The LMLP rollout is deliberately incremental and non-destructive. Each phase ends with a green build and a proof of equivalence on existing models before the next begins:

1. **Protocol skeleton** — the `Signal` / `Node` / `Edge` contracts land without touching the current pipeline.
2. **Unified container descriptor** — read support for `lmlp.graph` / `lmlp.tensors`.
3. **Legacy adapter** — existing `.rnn` files are mapped onto the LMLP graph, with forward equivalence proven.
4. **Unified scheduler** — a single scheduler driven only by each signal's propagation mode.
5. **Unified writing** — new `.rnn` files carry the `lmlp.*` description (alongside legacy blobs during transition).
6. **Consolidation** — the historical dense and LM code paths become topology builders, and legacy write paths are retired once equivalence holds everywhere.

Alongside this, the onboarding rollout (server setup and model packs) will be documented step by step:
- how to connect to the server setup,
- how to download/use provided models,
- how to validate outputs with existing tools,
- how to transition to custom training when needed.

## Documentation and Support

Alongside the release, I will publish step-by-step setup and usage documentation, including examples.

My intent is to make the path from "first clone" to "first useful result" much shorter and more reliable for everyone.
