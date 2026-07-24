use crate::base::math::Float;

fn log_softmax_in_place_impl<T: Float>(scores: &mut [T]) -> bool {
    if scores.is_empty() {
        return false;
    }
    let mut max_v = scores[0];
    for &v in scores.iter().skip(1) {
        if v > max_v {
            max_v = v;
        }
    }
    let mut sum = T::ZERO;
    for s in scores.iter_mut() {
        *s = (*s - max_v).exp();
        sum += *s;
    }
    if !sum.is_finite() || sum <= T::ZERO {
        return false;
    }
    let log_z = sum.ln();
    for s in scores.iter_mut() {
        *s = s.ln() - log_z;
    }
    true
}

pub fn log_softmax_in_place_f32(scores: &mut [f32]) -> bool {
    log_softmax_in_place_impl(scores)
}

pub fn log_softmax_in_place_f64(scores: &mut [f64]) -> bool {
    log_softmax_in_place_impl(scores)
}
