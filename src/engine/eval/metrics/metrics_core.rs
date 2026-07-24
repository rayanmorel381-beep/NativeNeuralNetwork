use crate::base::math::Float;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricError {
    Empty,
    ShapeMismatch,
    InvalidProbabilities,
}

fn mse_impl<T: Float>(prediction: &[T], target: &[T]) -> Result<T, MetricError> {
    if prediction.is_empty() {
        return Err(MetricError::Empty);
    }
    if prediction.len() != target.len() {
        return Err(MetricError::ShapeMismatch);
    }

    let mut acc = T::ZERO;
    for i in 0..prediction.len() {
        let d = prediction[i] - target[i];
        acc += d * d;
    }
    Ok(acc / T::from_usize(prediction.len()))
}

fn mae_impl<T: Float>(prediction: &[T], target: &[T]) -> Result<T, MetricError> {
    if prediction.is_empty() {
        return Err(MetricError::Empty);
    }
    if prediction.len() != target.len() {
        return Err(MetricError::ShapeMismatch);
    }

    let mut acc = T::ZERO;
    for i in 0..prediction.len() {
        acc += (prediction[i] - target[i]).abs();
    }
    Ok(acc / T::from_usize(prediction.len()))
}

fn argmax_impl<T: Float>(values: &[T]) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_value = values[0];
    for (idx, value) in values.iter().enumerate().skip(1) {
        if *value > best_value {
            best_value = *value;
            best_idx = idx;
        }
    }
    Some(best_idx)
}

fn accuracy_top1_from_one_hot_impl<T: Float>(
    prediction: &[T],
    one_hot_target: &[T],
) -> Result<T, MetricError> {
    if prediction.is_empty() {
        return Err(MetricError::Empty);
    }
    if prediction.len() != one_hot_target.len() {
        return Err(MetricError::ShapeMismatch);
    }

    let pred_idx = argmax_impl(prediction).ok_or(MetricError::Empty)?;
    let target_idx = argmax_impl(one_hot_target).ok_or(MetricError::Empty)?;
    Ok(if pred_idx == target_idx { T::ONE } else { T::ZERO })
}

fn cross_entropy_from_probabilities_impl<T: Float>(
    probabilities: &[T],
    one_hot_target: &[T],
    eps: T,
) -> Result<T, MetricError> {
    if probabilities.is_empty() {
        return Err(MetricError::Empty);
    }
    if probabilities.len() != one_hot_target.len() {
        return Err(MetricError::ShapeMismatch);
    }
    if !eps.is_finite() || eps <= T::ZERO {
        return Err(MetricError::InvalidProbabilities);
    }

    let mut loss = T::ZERO;
    for i in 0..probabilities.len() {
        let p = if probabilities[i] < eps {
            eps
        } else {
            probabilities[i]
        };
        if !p.is_finite() || p <= T::ZERO {
            return Err(MetricError::InvalidProbabilities);
        }
        loss -= one_hot_target[i] * p.ln();
    }
    Ok(loss)
}

pub fn mse_f32(prediction: &[f32], target: &[f32]) -> Result<f32, MetricError> {
    mse_impl(prediction, target)
}

pub fn mse_f64(prediction: &[f64], target: &[f64]) -> Result<f64, MetricError> {
    mse_impl(prediction, target)
}

pub fn mae_f32(prediction: &[f32], target: &[f32]) -> Result<f32, MetricError> {
    mae_impl(prediction, target)
}

pub fn mae_f64(prediction: &[f64], target: &[f64]) -> Result<f64, MetricError> {
    mae_impl(prediction, target)
}

pub fn argmax_f32(values: &[f32]) -> Option<usize> {
    argmax_impl(values)
}

pub fn argmax_f64(values: &[f64]) -> Option<usize> {
    argmax_impl(values)
}

pub fn accuracy_top1_from_one_hot_f32(
    prediction: &[f32],
    one_hot_target: &[f32],
) -> Result<f32, MetricError> {
    accuracy_top1_from_one_hot_impl(prediction, one_hot_target)
}

pub fn accuracy_top1_from_one_hot_f64(
    prediction: &[f64],
    one_hot_target: &[f64],
) -> Result<f64, MetricError> {
    accuracy_top1_from_one_hot_impl(prediction, one_hot_target)
}

pub fn cross_entropy_from_probabilities_f32(
    probabilities: &[f32],
    one_hot_target: &[f32],
    eps: f32,
) -> Result<f32, MetricError> {
    cross_entropy_from_probabilities_impl(probabilities, one_hot_target, eps)
}

pub fn cross_entropy_from_probabilities_f64(
    probabilities: &[f64],
    one_hot_target: &[f64],
    eps: f64,
) -> Result<f64, MetricError> {
    cross_entropy_from_probabilities_impl(probabilities, one_hot_target, eps)
}
