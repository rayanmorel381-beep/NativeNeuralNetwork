use crate::base::activations::{derivative_is_finite_f64, validate_activation_outputs_f64, ActivationKind};
use crate::graph::net::layers::{build_from_layers, LayerPlan, LayerSpec};
use crate::engine::eval::losses::{loss_and_gradient_f32, loss_and_gradient_f64, LossError, LossKind};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdConfig {
    pub learning_rate: f32,
    pub hidden_activation: ActivationKind,
    pub output_activation: ActivationKind,
    pub loss: LossKind,
    pub gradient_clip: Option<f32>,
}

impl SgdConfig {
    pub const fn new(
        learning_rate: f32,
        hidden_activation: ActivationKind,
        output_activation: ActivationKind,
        loss: LossKind,
    ) -> Self {
        Self {
            learning_rate,
            hidden_activation,
            output_activation,
            loss,
            gradient_clip: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainError {
    InvalidShape,
    InvalidConfig,
    CountMismatch,
    BufferTooSmall,
    ForwardNaN,
    LossError,
}

pub fn required_train_buffer_len(layers: &[usize]) -> Option<usize> {
    if layers.is_empty() {
        return None;
    }
    let mut total = 0usize;
    for &size in layers {
        total = total.checked_add(size)?;
    }
    Some(total)
}

pub struct SgdScratch<'a> {
    pub layer_specs_scratch: &'a mut [LayerSpec],
    pub activations_scratch: &'a mut [f32],
    pub deltas_scratch: &'a mut [f32],
    pub input_grad: Option<&'a mut [f32]>,
}

pub fn sgd_step(
    layers: &[usize],
    weights: &mut [f32],
    biases: &mut [f32],
    input: &[f32],
    target: &[f32],
    scratch: &mut SgdScratch,
    config: SgdConfig,
) -> Result<f32, TrainError> {
    let input_grad = scratch.input_grad.take();
    if !crate::engine::train::trainer::config_is_valid(config) {
        return Err(TrainError::InvalidConfig);
    }
    if !crate::engine::train::trainer::io_matches_layers(layers, input.len(), target.len()) {
        return Err(TrainError::InvalidShape);
    }

    let layer_count = build_from_layers(
        layers,
        config.hidden_activation,
        config.output_activation,
        weights.len(),
        biases.len(),
        scratch.layer_specs_scratch,
    )
    .map_err(map_layer_error)?;

    let layer_specs = &scratch.layer_specs_scratch[..layer_count];

    let mut layer_offsets = [0usize; 128];
    if layers.len() > layer_offsets.len() {
        return Err(TrainError::BufferTooSmall);
    }
    let mut running = 0usize;
    for (i, &size) in layers.iter().enumerate() {
        layer_offsets[i] = running;
        running = running
            .checked_add(size)
            .ok_or(TrainError::BufferTooSmall)?;
    }

    let required = LayerPlan {
        layers: layer_specs,
        weights,
        biases,
    }
    .total_neurons()
    .ok_or(TrainError::BufferTooSmall)?;
    if running != required {
        return Err(TrainError::BufferTooSmall);
    }
    if scratch.activations_scratch.len() < required || scratch.deltas_scratch.len() < required {
        return Err(TrainError::BufferTooSmall);
    }

    scratch.activations_scratch[..layers[0]].copy_from_slice(input);

    for (layer_idx, spec) in layer_specs.iter().enumerate() {
        let LayerSpec::Dense(dense) = *spec;

        let prev_off = layer_offsets[layer_idx];
        let curr_off = layer_offsets[layer_idx + 1];
        let (left, right) = scratch.activations_scratch.split_at_mut(curr_off);
        let prev = &left[prev_off..prev_off + dense.input_size];
        let curr = &mut right[..dense.output_size];

        let w_len = dense
            .input_size
            .checked_mul(dense.output_size)
            .ok_or(TrainError::InvalidShape)?;
        let w = &weights[dense.weight_offset..dense.weight_offset + w_len];
        let b = &biases[dense.bias_offset..dense.bias_offset + dense.output_size];

        forward_one(
            prev,
            curr,
            w,
            b,
            dense.input_size,
            dense.output_size,
            dense.activation,
        )?;
    }

    let out_idx = layers.len() - 1;
    let out_off = layer_offsets[out_idx];
    let out_size = layers[out_idx];

    let out_activations = &scratch.activations_scratch[out_off..out_off + out_size];
    let out_deltas = &mut scratch.deltas_scratch[out_off..out_off + out_size];

    let mut loss_grad = [0.0f32; 4096];
    if out_size > loss_grad.len() {
        return Err(TrainError::BufferTooSmall);
    }
    let loss = loss_and_gradient_f32(
        config.loss,
        out_activations,
        target,
        &mut loss_grad[..out_size],
    )
    .map_err(map_loss_error)?;

    let LayerSpec::Dense(last) = layer_specs[layer_count - 1];
    let output_activation = last.activation;

    for i in 0..out_size {
        let deriv = output_activation.derivative_from_output(out_activations[i]);
        out_deltas[i] = loss_grad[i] * deriv;
    }

    for rev in 1..layer_count {
        let curr_idx = layer_count - 1 - rev;
        let LayerSpec::Dense(curr_spec) = layer_specs[curr_idx];
        let LayerSpec::Dense(next_spec) = layer_specs[curr_idx + 1];

        let curr_off = layer_offsets[curr_idx + 1];
        let next_off = layer_offsets[curr_idx + 2];

        let curr_out_size = curr_spec.output_size;
        let next_out_size = next_spec.output_size;

        let (left_d, right_d) = scratch.deltas_scratch.split_at_mut(next_off);
        let curr_acts = &scratch.activations_scratch[curr_off..curr_off + curr_out_size];
        let next_deltas = &right_d[..next_out_size];
        let curr_deltas = &mut left_d[curr_off..curr_off + curr_out_size];

        let next_weights_len = next_spec
            .input_size
            .checked_mul(next_spec.output_size)
            .ok_or(TrainError::InvalidShape)?;
        let next_weights =
            &weights[next_spec.weight_offset..next_spec.weight_offset + next_weights_len];

        for i in 0..curr_out_size {
            let mut sum = 0.0f32;
            for o in 0..next_out_size {
                let w = next_weights[o * curr_out_size + i];
                sum += w * next_deltas[o];
            }
            let deriv = curr_spec.activation.derivative_from_output(curr_acts[i]);
            curr_deltas[i] = sum * deriv;
        }
    }

    if let Some(ig) = input_grad {
        let LayerSpec::Dense(first) = layer_specs[0];
        let in_size = first.input_size;
        let out_size = first.output_size;
        if ig.len() >= in_size {
            let w_len = in_size
                .checked_mul(out_size)
                .ok_or(TrainError::InvalidShape)?;
            let w0 = &weights[first.weight_offset..first.weight_offset + w_len];
            let delta1 = &scratch.deltas_scratch[layer_offsets[1]..layer_offsets[1] + out_size];
            for j in 0..in_size {
                let mut sum = 0.0f32;
                for o in 0..out_size {
                    sum += w0[o * in_size + j] * delta1[o];
                }
                ig[j] = sum;
            }
        }
    }

    for (layer_idx, spec) in layer_specs.iter().enumerate() {
        let LayerSpec::Dense(dense) = *spec;

        let prev_off = layer_offsets[layer_idx];
        let curr_off = layer_offsets[layer_idx + 1];
        let prev = &scratch.activations_scratch[prev_off..prev_off + dense.input_size];
        let curr_delta = &scratch.deltas_scratch[curr_off..curr_off + dense.output_size];

        let w_len = dense
            .input_size
            .checked_mul(dense.output_size)
            .ok_or(TrainError::InvalidShape)?;
        let w = &mut weights[dense.weight_offset..dense.weight_offset + w_len];
        let b = &mut biases[dense.bias_offset..dense.bias_offset + dense.output_size];

        let args = ApplySgdArgs {
            weights: w,
            biases: b,
            prev_activation: prev,
            delta: curr_delta,
            in_size: dense.input_size,
            out_size: dense.output_size,
            learning_rate: config.learning_rate,
            clip: config.gradient_clip,
        };
        apply_sgd_update(args)?;
    }

    Ok(loss)
}

fn forward_one(
    input: &[f32],
    output: &mut [f32],
    weights: &[f32],
    biases: &[f32],
    in_size: usize,
    out_size: usize,
    activation: ActivationKind,
) -> Result<(), TrainError> {
    match activation {
        ActivationKind::Relu => {
            let ro_w = crate::base::tensor::TensorViewRo::from_slice(weights, [1, 1, 1, out_size, in_size]);
            if ro_w.len() < out_size * in_size { return Err(TrainError::ForwardNaN); }
            if ro_w.is_empty() { return Err(TrainError::ForwardNaN); }
            if ro_w.checksum() == 0 { return Err(TrainError::ForwardNaN); }
            for o in 0..out_size {
                let row = o * in_size;
                let mut acc = 0.0f32;
                for i in 0..in_size {
                    acc += ro_w.data[row + i] * input[i];
                }
                output[o] = acc;
            }
            let mut tv = crate::base::tensor::TensorView {
                data: output,
                shape: [1, 1, 1, out_size, 1],
            };
            crate::base::tensor::tensor_add_bias(&mut tv, biases);
            crate::base::tensor::tensor_relu(&mut tv);
            for y in tv.data[..out_size].iter() {
                if !y.is_finite() {
                    return Err(TrainError::ForwardNaN);
                }
            }
            Ok(())
        }
        ActivationKind::SoftmaxOutput
            if in_size <= 64 && in_size * out_size <= 4096 =>
        {
            let mut w_copy = [0.0f32; 4096];
            let mut inp_copy = [0.0f32; 64];
            let wlen = in_size * out_size;
            w_copy[..wlen].copy_from_slice(&weights[..wlen]);
            inp_copy[..in_size].copy_from_slice(&input[..in_size]);
            {
                let tw = crate::base::tensor::TensorView {
                    data: &mut w_copy[..wlen],
                    shape: [1, 1, 1, out_size, in_size],
                };
                let ti = crate::base::tensor::TensorView {
                    data: &mut inp_copy[..in_size],
                    shape: [1, 1, 1, in_size, 1],
                };
                let mut tout = crate::base::tensor::TensorView {
                    data: &mut output[..out_size],
                    shape: [1, 1, 1, out_size, 1],
                };
                crate::base::tensor::tensor_matmul_add(&tw, &ti, &mut tout, out_size, in_size, 1);
                let _ = ti;
            }
            let mut tv = crate::base::tensor::TensorView {
                data: &mut output[..out_size],
                shape: [1, 1, 1, out_size, 1],
            };
            crate::base::tensor::tensor_add_bias(&mut tv, biases);
            crate::base::tensor::tensor_softmax_rows(&mut tv);
            for y in tv.data[..out_size].iter() {
                if !y.is_finite() {
                    return Err(TrainError::ForwardNaN);
                }
            }
            Ok(())
        }
        _ => {
            for o in 0..out_size {
                let row = o * in_size;
                let mut acc = biases[o];
                for i in 0..in_size {
                    acc += weights[row + i] * input[i];
                }
                let y = activation.apply(acc);
                if !y.is_finite() {
                    return Err(TrainError::ForwardNaN);
                }
                output[o] = y;
            }
            Ok(())
        }
    }
}

pub struct ApplySgdArgs<'a> {
    pub weights: &'a mut [f32],
    pub biases: &'a mut [f32],
    pub prev_activation: &'a [f32],
    pub delta: &'a [f32],
    pub in_size: usize,
    pub out_size: usize,
    pub learning_rate: f32,
    pub clip: Option<f32>,
}

fn apply_sgd_update(args: ApplySgdArgs) -> Result<(), TrainError> {
    let ApplySgdArgs {
        weights,
        biases,
        prev_activation,
        delta,
        in_size,
        out_size,
        learning_rate,
        clip,
    } = args;

    let kind = crate::engine::train::optimizers::OptimizerKind::Sgd { momentum: 0.0f32, nesterov: false };
    let mut grad_row = [0.0f32; 4096];
    let mut velocity_row = [0.0f32; 4096];
    if in_size > grad_row.len() {
        return Err(TrainError::BufferTooSmall);
    }
    let state_len =
        crate::engine::train::optimizers::optimizer_state_len(kind, in_size).ok_or(TrainError::BufferTooSmall)?;
    if velocity_row.len() < state_len {
        return Err(TrainError::BufferTooSmall);
    }

    for o in 0..out_size {
        let row = o * in_size;
        for i in 0..in_size {
            grad_row[i] = delta[o] * prev_activation[i];
        }
        let grad = &mut grad_row[..in_size];
        if crate::engine::train::gradients::has_nan_f32(grad) || crate::engine::train::gradients::has_inf_f32(grad) {
            return Err(TrainError::ForwardNaN);
        }
        {
            let mut tt = crate::base::tensor::TrainingTensor::from_slices(
                &mut weights[row..row + in_size],
                &mut velocity_row[..in_size],
                [1, 1, 1, 1, in_size],
            );
            if tt.numel() != in_size || tt.data.len() != in_size || tt.shape[4] != in_size {
                return Err(TrainError::InvalidConfig);
            }
            tt.zero_grad();
            tt.accumulate_grad(grad);
            tt.apply_grad(0.0f32);
        }
        if let Some(limit) = clip {
            crate::engine::train::gradients::clip_by_global_norm(grad, limit)
                .map_err(|_| TrainError::InvalidConfig)?;
            if !crate::engine::train::gradients::within_abs_bound_f32(grad, limit) {
                return Err(TrainError::InvalidConfig);
            }
        }
        crate::engine::train::optimizers::apply_optimizer_step(
            kind,
            &mut weights[row..row + in_size],
            grad,
            &mut velocity_row[..state_len],
            learning_rate,
            0,
        )
        .map_err(|_| TrainError::InvalidConfig)?;

        let mut grad_b = [delta[o]];
        if !crate::engine::train::gradients::all_finite(&grad_b) {
            return Err(TrainError::ForwardNaN);
        }
        if let Some(limit) = clip {
            crate::engine::train::gradients::clip_by_global_norm(&mut grad_b, limit)
                .map_err(|_| TrainError::InvalidConfig)?;
        }
        let mut vel_b = [0.0f32];
        crate::engine::train::optimizers::apply_optimizer_step(
            kind,
            &mut biases[o..o + 1],
            &grad_b,
            &mut vel_b,
            learning_rate,
            0,
        )
        .map_err(|_| TrainError::InvalidConfig)?;
    }
    Ok(())
}

fn map_layer_error(err: crate::graph::net::layers::LayerError) -> TrainError {
    match err {
        crate::graph::net::layers::LayerError::EmptyPlan => TrainError::CountMismatch,
        crate::graph::net::layers::LayerError::InvalidShape => TrainError::CountMismatch,
        crate::graph::net::layers::LayerError::InvalidRange => TrainError::CountMismatch,
        crate::graph::net::layers::LayerError::IncompatibleChain => TrainError::CountMismatch,
        crate::graph::net::layers::LayerError::BufferTooSmall => TrainError::CountMismatch,
        crate::graph::net::layers::LayerError::CountMismatch => TrainError::CountMismatch,
    }
}

fn map_loss_error(err: LossError) -> TrainError {
    match err {
        LossError::Empty => TrainError::LossError,
        LossError::ShapeMismatch => TrainError::LossError,
        LossError::NonFinite => TrainError::LossError,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgdConfigF64 {
    pub learning_rate: f64,
    pub hidden_activation: ActivationKind,
    pub output_activation: ActivationKind,
    pub loss: LossKind,
    pub gradient_clip: Option<f64>,
}

pub struct SgdScratchF64<'a> {
    pub layer_specs_scratch: &'a mut [LayerSpec],
    pub activations_scratch: &'a mut [f64],
    pub deltas_scratch: &'a mut [f64],
}

pub fn sgd_step_f64(
    layers: &[usize],
    weights: &mut [f64],
    biases: &mut [f64],
    input: &[f64],
    target: &[f64],
    scratch: &mut SgdScratchF64,
    config: SgdConfigF64,
) -> Result<f64, TrainError> {
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(TrainError::InvalidConfig);
    }
    if !crate::engine::train::trainer::io_matches_layers(layers, input.len(), target.len()) {
        return Err(TrainError::InvalidShape);
    }

    let layer_count = build_from_layers(
        layers,
        config.hidden_activation,
        config.output_activation,
        weights.len(),
        biases.len(),
        scratch.layer_specs_scratch,
    )
    .map_err(map_layer_error)?;

    let layer_specs = &scratch.layer_specs_scratch[..layer_count];

    let mut layer_offsets = [0usize; 128];
    if layers.len() > layer_offsets.len() {
        return Err(TrainError::BufferTooSmall);
    }
    let mut running = 0usize;
    for (i, &size) in layers.iter().enumerate() {
        layer_offsets[i] = running;
        running = running
            .checked_add(size)
            .ok_or(TrainError::BufferTooSmall)?;
    }

    let required = running;
    if scratch.activations_scratch.len() < required || scratch.deltas_scratch.len() < required {
        return Err(TrainError::BufferTooSmall);
    }

    scratch.activations_scratch[..layers[0]].copy_from_slice(input);

    for (layer_idx, spec) in layer_specs.iter().enumerate() {
        let LayerSpec::Dense(dense) = *spec;

        let prev_off = layer_offsets[layer_idx];
        let curr_off = layer_offsets[layer_idx + 1];
        let (left, right) = scratch.activations_scratch.split_at_mut(curr_off);
        let prev = &left[prev_off..prev_off + dense.input_size];
        let curr = &mut right[..dense.output_size];

        let w_len = dense
            .input_size
            .checked_mul(dense.output_size)
            .ok_or(TrainError::InvalidShape)?;
        let w = &weights[dense.weight_offset..dense.weight_offset + w_len];
        let b = &biases[dense.bias_offset..dense.bias_offset + dense.output_size];

        for o in 0..dense.output_size {
            let row = o * dense.input_size;
            let mut acc = b[o];
            for i in 0..dense.input_size {
                acc += w[row + i] * prev[i];
            }
            curr[o] = acc;
        }
        if !validate_activation_outputs_f64(dense.activation, curr) {
            return Err(TrainError::ForwardNaN);
        }
        dense.activation.apply_in_place_f64(curr);
    }

    let out_idx = layers.len() - 1;
    let out_off = layer_offsets[out_idx];
    let out_size = layers[out_idx];

    let out_activations = &scratch.activations_scratch[out_off..out_off + out_size];
    let out_deltas = &mut scratch.deltas_scratch[out_off..out_off + out_size];

    let mut loss_grad = [0.0f64; 4096];
    if out_size > loss_grad.len() {
        return Err(TrainError::BufferTooSmall);
    }
    let loss = loss_and_gradient_f64(
        config.loss,
        out_activations,
        target,
        &mut loss_grad[..out_size],
    )
    .map_err(map_loss_error)?;

    let LayerSpec::Dense(last) = layer_specs[layer_count - 1];
    let output_activation = last.activation;

    if !derivative_is_finite_f64(output_activation, out_activations) {
        return Err(TrainError::ForwardNaN);
    }
    for i in 0..out_size {
        let deriv = output_activation.derivative_from_output_f64(out_activations[i]);
        out_deltas[i] = loss_grad[i] * deriv;
    }

    for rev in 1..layer_count {
        let curr_idx = layer_count - 1 - rev;
        let LayerSpec::Dense(curr_spec) = layer_specs[curr_idx];
        let LayerSpec::Dense(next_spec) = layer_specs[curr_idx + 1];

        let curr_off = layer_offsets[curr_idx + 1];
        let next_off = layer_offsets[curr_idx + 2];

        let curr_out_size = curr_spec.output_size;
        let next_out_size = next_spec.output_size;

        let (left_d, right_d) = scratch.deltas_scratch.split_at_mut(next_off);
        let curr_acts = &scratch.activations_scratch[curr_off..curr_off + curr_out_size];
        let next_deltas = &right_d[..next_out_size];
        let curr_deltas = &mut left_d[curr_off..curr_off + curr_out_size];

        let next_weights_len = next_spec
            .input_size
            .checked_mul(next_spec.output_size)
            .ok_or(TrainError::InvalidShape)?;
        let next_weights =
            &weights[next_spec.weight_offset..next_spec.weight_offset + next_weights_len];

        for i in 0..curr_out_size {
            let mut sum = 0.0f64;
            for o in 0..next_out_size {
                let w = next_weights[o * curr_out_size + i];
                sum += w * next_deltas[o];
            }
            let deriv = curr_spec.activation.derivative_from_output_f64(curr_acts[i]);
            curr_deltas[i] = sum * deriv;
        }
    }

    for (layer_idx, spec) in layer_specs.iter().enumerate() {
        let LayerSpec::Dense(dense) = *spec;

        let prev_off = layer_offsets[layer_idx];
        let curr_off = layer_offsets[layer_idx + 1];
        let prev = &scratch.activations_scratch[prev_off..prev_off + dense.input_size];
        let curr_delta = &scratch.deltas_scratch[curr_off..curr_off + dense.output_size];

        let w_len = dense
            .input_size
            .checked_mul(dense.output_size)
            .ok_or(TrainError::InvalidShape)?;
        let w = &mut weights[dense.weight_offset..dense.weight_offset + w_len];
        let b = &mut biases[dense.bias_offset..dense.bias_offset + dense.output_size];

        apply_sgd_update_f64(ApplySgdArgsF64 {
            weights: w,
            biases: b,
            prev_activation: prev,
            delta: curr_delta,
            in_size: dense.input_size,
            out_size: dense.output_size,
            learning_rate: config.learning_rate,
            clip: config.gradient_clip,
        })?;
    }

    Ok(loss)
}

pub struct ApplySgdArgsF64<'a> {
    pub weights: &'a mut [f64],
    pub biases: &'a mut [f64],
    pub prev_activation: &'a [f64],
    pub delta: &'a [f64],
    pub in_size: usize,
    pub out_size: usize,
    pub learning_rate: f64,
    pub clip: Option<f64>,
}

fn apply_sgd_update_f64(args: ApplySgdArgsF64) -> Result<(), TrainError> {
    let ApplySgdArgsF64 {
        weights,
        biases,
        prev_activation,
        delta,
        in_size,
        out_size,
        learning_rate,
        clip,
    } = args;
    let kind = crate::engine::train::optimizers::OptimizerKind::Sgd { momentum: 0.0f64, nesterov: false };
    let mut grad_row = [0.0f64; 4096];
    let mut velocity_row = [0.0f64; 4096];
    if in_size > grad_row.len() {
        return Err(TrainError::BufferTooSmall);
    }
    let state_len =
        crate::engine::train::optimizers::optimizer_state_len(kind, in_size).ok_or(TrainError::BufferTooSmall)?;
    if velocity_row.len() < state_len {
        return Err(TrainError::BufferTooSmall);
    }

    for o in 0..out_size {
        let row = o * in_size;
        for i in 0..in_size {
            grad_row[i] = delta[o] * prev_activation[i];
        }
        let grad = &mut grad_row[..in_size];
        if crate::engine::train::gradients::has_nan_f64(grad) || crate::engine::train::gradients::has_inf_f64(grad) {
            return Err(TrainError::ForwardNaN);
        }
        if let Some(limit) = clip {
            crate::engine::train::gradients::clip_by_global_norm(grad, limit)
                .map_err(|_| TrainError::InvalidConfig)?;
            if !crate::engine::train::gradients::within_abs_bound_f64(grad, limit) {
                return Err(TrainError::InvalidConfig);
            }
        }
        crate::engine::train::optimizers::apply_optimizer_step(
            kind,
            &mut weights[row..row + in_size],
            grad,
            &mut velocity_row[..state_len],
            learning_rate,
            0,
        )
        .map_err(|_| TrainError::InvalidConfig)?;

        let mut grad_b = [delta[o]];
        if !crate::engine::train::gradients::all_finite(&grad_b) {
            return Err(TrainError::ForwardNaN);
        }
        if let Some(limit) = clip {
            crate::engine::train::gradients::clip_by_global_norm(&mut grad_b, limit)
                .map_err(|_| TrainError::InvalidConfig)?;
        }
        let mut vel_b = [0.0f64];
        crate::engine::train::optimizers::apply_optimizer_step(
            kind,
            &mut biases[o..o + 1],
            &grad_b,
            &mut vel_b,
            learning_rate,
            0,
        )
        .map_err(|_| TrainError::InvalidConfig)?;
    }
    Ok(())
}
