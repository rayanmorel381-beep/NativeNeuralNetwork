use crate::graph::net::layers::LayerSpec;
use crate::graph::lm::LmError;
use crate::base::math::Float;
use crate::graph::propagation::TransformCtx;
use super::scratch::ForwardScratch;

pub(crate) fn mlp_graph_forward<T: Float>(
    specs: &[LayerSpec],
    weights: &[T],
    biases: &[T],
    input: &[T],
    hidden: &mut [T],
    norm_buf: &mut [T],
    out: &mut [T],
) -> Result<(), LmError> {
    let ctx = TransformCtx::dense(weights, biases);
    let mut sc = ForwardScratch {
        hidden,
        norm_buf,
        qkv: &mut [],
        wo_out: &mut [],
        scores: &mut [],
        attn_head: &mut [],
        ffn_gate: &mut [],
        ffn_up: &mut [],
        ffn_out: &mut [],
        logits: &mut [],
        head_z1: &mut [],
        head_a1: &mut [],
        moe_aux_loss: 0.0,
    };
    let in0 = match specs.first() {
        Some(LayerSpec::Dense(d)) => d.input_size,
        None => return Err(LmError::ShapeMismatch),
    };
    if input.len() < in0 || sc.hidden.len() < in0 {
        return Err(LmError::ShapeMismatch);
    }
    sc.hidden[..in0].copy_from_slice(&input[..in0]);
    crate::graph::scheduler::run_lmlp(
        &ctx,
        crate::graph::topology::LmlpTopology::Dense(specs),
        &mut [],
        &mut sc,
    )?;
    let out_size = match specs.last() {
        Some(LayerSpec::Dense(d)) => d.output_size,
        None => return Err(LmError::ShapeMismatch),
    };
    if out.len() < out_size || sc.hidden.len() < out_size {
        return Err(LmError::ShapeMismatch);
    }
    out[..out_size].copy_from_slice(&sc.hidden[..out_size]);
    Ok(())
}
