# transformer/beam_search

Beam search decoding: keep the top-N partial sequences at each step instead of
committing to a single greedy token.

Façade: `private_api::modules::beam_search`.

## Key types

- `BeamError` — failure reason.

## Functions

- `select_top_beams(...)` — keep the highest-scoring beams.
- `log_softmax_in_place_f32(...)` / `log_softmax_in_place_f64(...)` — convert
  logits to log-probabilities for stable score accumulation.

## Concrete usage

Score candidates in log space and keep the best beams each step:

```rust
use native_neural_network::private_api::modules::beam_search;

fn expand(logits: &mut [f32], beam_scores: &mut [f32], width: usize) {
    beam_search::log_softmax_in_place_f32(logits);   // stable log-probs
    // add token log-probs to running beam scores, then prune:
    beam_search::select_top_beams(beam_scores, width);
}
```

## Performance levers

- Beam width is the main cost/quality knob — width 1 degrades to greedy.
- Accumulate scores in log space (`log_softmax_in_place_*`) to avoid underflow
  over long sequences.

## Integration

Driven by [inference](../transformer/inference.md) `generate_beam`; an
alternative to plain [sampling](../transformer/sampling.md).
