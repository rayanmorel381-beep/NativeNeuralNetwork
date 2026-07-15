# transformer/sampling

Token sampling strategies for generation: temperature, top-k, top-p (nucleus)
and greedy argmax.

Façade: `private_api::modules::sampling`.

## Key types

- `SamplingError` — failure reason.

## Functions

- `softmax_temperature(...)` — softmax with a temperature scale.
- `top_k_mask(...)` — zero out all but the k highest logits.
- `top_p_cutoff(...)` — nucleus threshold over sorted cumulative probabilities.
- `argmax_sample(...)` — greedy pick.
- `sample_from_cumulative(...)` — draw from a cumulative distribution.

## Concrete usage

A typical nucleus-sampling pipeline: temperature → top-k → top-p → draw:

```rust
use native_neural_network::private_api::modules::sampling;

fn sample(logits: &mut [f32], temp: f32, k: usize, p: f32, u: f32) -> usize {
    sampling::softmax_temperature(logits, temp);   // sharpen/flatten
    sampling::top_k_mask(logits, k);               // keep k best
    let cutoff = sampling::top_p_cutoff(logits, p);// nucleus threshold
    sampling::sample_from_cumulative(logits, cutoff, u) // u in [0,1)
}

// Deterministic decoding:
fn greedy(logits: &[f32]) -> usize {
    sampling::argmax_sample(logits)
}
```

## Performance levers

- Lower temperature → sharper, more deterministic; higher → more diverse.
- `top_k_mask` before `top_p_cutoff` shrinks the candidate set cheaply.
- Use `argmax_sample` when you need reproducible output (no RNG draw).

## Integration

Consumes logits from [embeddings](../transformer/embeddings.md) MLP head, driven
by [inference](../transformer/inference.md); an alternative to
[beam_search](../transformer/beam_search.md).
