use super::errors::MoeError;
use crate::base::math::Float;

pub fn top1_gating(scores: &[f32], num_experts: usize) -> Result<usize, MoeError> {
    if scores.is_empty() || num_experts == 0 || scores.len() < num_experts {
        return Err(MoeError::Empty);
    }
    let mut best_idx = 0usize;
    let mut best = scores[0];
    for (i, &v) in scores.iter().enumerate().take(num_experts).skip(1) {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    Ok(best_idx)
}

pub fn softmax_scores_inplace(scores: &mut [f32], num_experts: usize) -> Result<(), MoeError> {
    if scores.is_empty() || num_experts == 0 || scores.len() < num_experts {
        return Err(MoeError::Empty);
    }
    let s = &mut scores[..num_experts];
    let max = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in s.iter_mut() {
        *v = Float::exp(*v - max);
        sum += *v;
    }
    if sum > 0.0 {
        for v in s.iter_mut() {
            *v /= sum;
        }
    }
    Ok(())
}

pub fn top2_gating_weighted(
    scores: &[f32],
    num_experts: usize,
) -> Result<(usize, usize, f32, f32), MoeError> {
    if scores.len() < num_experts || num_experts < 2 {
        return Err(MoeError::Empty);
    }
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = (1usize, f32::NEG_INFINITY);
    for (i, &v) in scores.iter().enumerate().take(num_experts) {
        if v > best.1 {
            second = best;
            best = (i, v);
        } else if v > second.1 {
            second = (i, v);
        }
    }
    let total = best.1 + second.1;
    let (w1, w2) = if total > 0.0 {
        (best.1 / total, second.1 / total)
    } else {
        (0.5, 0.5)
    };
    Ok((best.0, second.0, w1, w2))
}
