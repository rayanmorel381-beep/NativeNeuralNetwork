# transformer/moe

Mixture-of-experts routing. Sends each token to a small subset of experts,
scaling capacity without scaling per-token compute.

Façade: `private_api::modules::moe`.

## Key types

- `MoeError` — failure reason.

## Functions

- Routing: `route_top1(...)`, `route_top2_weighted(...)`, `top1_gating(...)`,
  `top2_gating_weighted(...)`.
- Scores: `softmax_scores_inplace(...)`.
- Load balancing: `moe_load_aux_loss(...)`.
- Router diagnostics: `moe_router_condition`, `moe_router_determinant`,
  `moe_router_eigenvectors`, `moe_router_inverse_gram`, `moe_router_is_pd`.

## Concrete usage

Route a token to its single best expert (cheapest path), and add the balancing
aux-loss during training so experts don't collapse:

```rust
use native_neural_network::private_api::modules::moe;

fn route(gate_logits: &mut [f32]) -> usize {
    moe::softmax_scores_inplace(gate_logits);
    moe::route_top1(gate_logits)          // index of the chosen expert
}

fn training_penalty(assignments: &[usize], n_experts: usize) -> f32 {
    moe::moe_load_aux_loss(assignments, n_experts)   // add to total loss
}
```

## Performance levers

- `route_top1` is cheapest (one expert/token); `route_top2_weighted` trades
  compute for quality.
- Add `moe_load_aux_loss` to the objective to keep experts balanced.
- Use `moe_router_condition` / `moe_router_is_pd` to catch an ill-conditioned
  router before it destabilizes training.

## Integration

Runs inside transformer blocks
([inference](../transformer/inference.md) `MoeBlockWeights` in
[lm](../transformer/lm.md)); the aux loss feeds [losses](../training/losses.md).
