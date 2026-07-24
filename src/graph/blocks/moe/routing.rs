use super::errors::MoeError;
use crate::base::math::Float;

pub fn route_top1<T: Float>(
    expert_out: &[T],
    h: usize,
    out: &mut [T],
) -> Result<(), MoeError> {
    if h == 0 || expert_out.len() < h || out.len() < h {
        return Err(MoeError::ShapeMismatch);
    }
    out[..h].copy_from_slice(&expert_out[..h]);
    Ok(())
}

pub fn route_top2_weighted<T: Float>(
    out1: &[T],
    out2: &[T],
    h: usize,
    w1: f32,
    w2: f32,
    out: &mut [T],
) -> Result<(), MoeError> {
    if h == 0 || out1.len() < h || out2.len() < h || out.len() < h {
        return Err(MoeError::ShapeMismatch);
    }
    let tw1 = T::from_f32(w1);
    let tw2 = T::from_f32(w2);
    for i in 0..h {
        out[i] = tw1 * out1[i] + tw2 * out2[i];
    }
    Ok(())
}

pub fn moe_load_aux_loss(
    expert_frac: &[f32],
    gate_prob_sum: &[f32],
    num_experts: usize,
) -> f32 {
    if num_experts == 0 || expert_frac.len() < num_experts || gate_prob_sum.len() < num_experts {
        return 0.0;
    }
    let n = num_experts as f32;
    let mut loss = 0.0f32;
    for e in 0..num_experts {
        loss += expert_frac[e] * gate_prob_sum[e];
    }
    loss * n
}
