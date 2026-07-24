use crate::base::activations::ActivationKind;
use crate::graph::net::layers::{build_from_layers, LayerSpec};
use crate::engine::train::trainer::{sgd_step, sgd_step_f64, required_train_buffer_len, SgdConfig, SgdConfigF64, SgdScratch, SgdScratchF64};
use crate::engine::rnn_flow::{
    bytes_as_f32_mut, bytes_as_f64_mut, RnnFlowError, MAX_LAYERS, MAX_TOTAL_NEURONS,
};
use super::entry::ParsedLayout;

pub(crate) fn sgd_on_decrypted(
    bytes: &mut [u8],
    samples: &[(&[f32], &[f32])],
    learning_rate: f32,
    gradient_clip: Option<f32>,
    layout: &ParsedLayout,
) -> Result<(f64, usize), RnnFlowError> {
    let topo = &layout.topology[..layout.topo_len];
    let topo_len = layout.topo_len;
    let dtype = layout.dtype;
    let w_start = layout.w_start;
    let w_end = layout.w_end;
    let b_start = layout.b_start;
    let b_end = layout.b_end;
    let hidden_activation = layout.hidden_activation;
    let output_activation = layout.output_activation;
    let buf_len = match required_train_buffer_len(topo) {
        Some(n) if n <= MAX_TOTAL_NEURONS => n,
        _ => return Err(RnnFlowError::BadBytes),
    };
    let lc = topo_len.saturating_sub(1);
    let mut loss_acc = crate::engine::eval::metrics::RunningMeanF64::new();
    match dtype {
        0 => {
            if !(w_end - w_start).is_multiple_of(4) || !(b_end - b_start).is_multiple_of(4) {
                return Err(RnnFlowError::BadBytes);
            }
            let mut layer_specs_buf = [LayerSpec::Dense(crate::graph::net::layers::LayerDesc {
                input_size: 1, output_size: 1, weight_offset: 0, bias_offset: 0,
                activation: ActivationKind::Identity,
            }); MAX_LAYERS];
            let mut activations_buf = [0.0f32; MAX_TOTAL_NEURONS];
            let mut deltas_buf = [0.0f32; MAX_TOTAL_NEURONS];
            let (weights, biases) = if w_start < b_start {
                if w_end > b_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(b_start);
                (bytes_as_f32_mut(&mut left[w_start..w_end]), bytes_as_f32_mut(&mut right[..b_end - b_start]))
            } else {
                if b_end > w_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(w_start);
                (bytes_as_f32_mut(&mut right[..w_end - w_start]), bytes_as_f32_mut(&mut left[b_start..b_end]))
            };
            let mut cfg = SgdConfig::new(
                learning_rate, hidden_activation, output_activation, layout.loss,
            );
            cfg.gradient_clip = gradient_clip;
            for &(input, target) in samples {
                if input.len() != topo[0] || target.len() != topo[topo_len - 1] { continue; }
                let mut scratch = SgdScratch {
                    layer_specs_scratch: &mut layer_specs_buf[..lc],
                    activations_scratch: &mut activations_buf[..buf_len],
                    deltas_scratch: &mut deltas_buf[..buf_len],
                    input_grad: None,
                };
                if let Ok(loss) = sgd_step(topo, weights, biases, input, target, &mut scratch, cfg) {
                    loss_acc.update(loss as f64);
                }
            }
        }
        1 => {
            if !(w_end - w_start).is_multiple_of(8) || !(b_end - b_start).is_multiple_of(8) {
                return Err(RnnFlowError::BadBytes);
            }
            let mut layer_specs_buf = [LayerSpec::Dense(crate::graph::net::layers::LayerDesc {
                input_size: 1, output_size: 1, weight_offset: 0, bias_offset: 0,
                activation: ActivationKind::Identity,
            }); MAX_LAYERS];
            let mut activations_buf = [0.0f64; MAX_TOTAL_NEURONS];
            let mut deltas_buf = [0.0f64; MAX_TOTAL_NEURONS];
            let mut input_f64 = [0.0f64; MAX_TOTAL_NEURONS];
            let mut target_f64 = [0.0f64; MAX_TOTAL_NEURONS];
            let (weights, biases) = if w_start < b_start {
                if w_end > b_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(b_start);
                (bytes_as_f64_mut(&mut left[w_start..w_end]), bytes_as_f64_mut(&mut right[..b_end - b_start]))
            } else {
                if b_end > w_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(w_start);
                (bytes_as_f64_mut(&mut right[..w_end - w_start]), bytes_as_f64_mut(&mut left[b_start..b_end]))
            };
            let cfg = SgdConfigF64 {
                learning_rate: learning_rate as f64, hidden_activation, output_activation,
                loss: layout.loss, gradient_clip: gradient_clip.map(|c| c as f64),
            };
            for &(input, target) in samples {
                if input.len() != topo[0] || target.len() != topo[topo_len - 1] { continue; }
                for (i, &v) in input.iter().enumerate() { input_f64[i] = v as f64; }
                for (i, &v) in target.iter().enumerate() { target_f64[i] = v as f64; }
                let mut scratch = SgdScratchF64 {
                    layer_specs_scratch: &mut layer_specs_buf[..lc],
                    activations_scratch: &mut activations_buf[..buf_len],
                    deltas_scratch: &mut deltas_buf[..buf_len],
                };
                if let Ok(loss) = sgd_step_f64(topo, weights, biases, &input_f64[..input.len()], &target_f64[..target.len()], &mut scratch, cfg) {
                    loss_acc.update(loss);
                }
            }
        }
        _ => return Err(RnnFlowError::BadBytes),
    }
    Ok((loss_acc.sum, loss_acc.count as usize))
}

pub(crate) struct EvalAccum {
    pub mse: crate::engine::eval::metrics::RunningMean,
    pub mae: crate::engine::eval::metrics::RunningMean,
    pub accuracy: crate::engine::eval::metrics::RunningMean,
    pub cross_entropy: crate::engine::eval::metrics::RunningMean,
    pub confidence: crate::engine::eval::metrics::RunningMean,
    pub ops: crate::observability::profiler::OpCounter,
}

impl EvalAccum {
    pub(crate) fn new() -> Self {
        Self {
            mse: crate::engine::eval::metrics::RunningMean::new(),
            mae: crate::engine::eval::metrics::RunningMean::new(),
            accuracy: crate::engine::eval::metrics::RunningMean::new(),
            cross_entropy: crate::engine::eval::metrics::RunningMean::new(),
            confidence: crate::engine::eval::metrics::RunningMean::new(),
            ops: crate::observability::profiler::OpCounter::default(),
        }
    }

    pub(crate) fn reset(&mut self) {
        self.mse = crate::engine::eval::metrics::RunningMean::default();
        self.mae = crate::engine::eval::metrics::RunningMean::default();
        self.accuracy = crate::engine::eval::metrics::RunningMean::default();
        self.cross_entropy = crate::engine::eval::metrics::RunningMean::default();
        self.confidence = crate::engine::eval::metrics::RunningMean::default();
        self.ops.reset();
    }

    pub(crate) fn merge(&mut self, other: &EvalAccum) {
        self.mse.merge(other.mse);
        self.mae.merge(other.mae);
        self.accuracy.merge(other.accuracy);
        self.cross_entropy.merge(other.cross_entropy);
        self.confidence.merge(other.confidence);
        self.ops.merge(&other.ops);
    }
}

fn eval_forward_f32(
    topo: &[usize],
    weights: &[f32],
    biases: &[f32],
    layer_specs: &[LayerSpec],
    input: &[f32],
    activations: &mut [f32],
    ops: &mut crate::observability::profiler::OpCounter,
) -> Option<(usize, usize)> {
    let mut offs = [0usize; MAX_LAYERS + 1];
    if topo.len() >= offs.len() {
        return None;
    }
    let mut running = 0usize;
    for (i, &size) in topo.iter().enumerate() {
        offs[i] = running;
        running = running.checked_add(size)?;
    }
    if activations.len() < running || input.len() != topo[0] {
        return None;
    }
    activations[..topo[0]].copy_from_slice(input);
    for (layer_idx, spec) in layer_specs.iter().enumerate() {
        let LayerSpec::Dense(dense) = *spec;
        let prev_off = offs[layer_idx];
        let curr_off = offs[layer_idx + 1];
        let (left, right) = activations.split_at_mut(curr_off);
        let prev = &left[prev_off..prev_off + dense.input_size];
        let curr = &mut right[..dense.output_size];
        let w_len = dense.input_size.checked_mul(dense.output_size)?;
        let w = &weights[dense.weight_offset..dense.weight_offset + w_len];
        let b = &biases[dense.bias_offset..dense.bias_offset + dense.output_size];
        for o in 0..dense.output_size {
            let row = o * dense.input_size;
            let mut acc = b[o];
            for i in 0..dense.input_size {
                acc += w[row + i] * prev[i];
            }
            curr[o] = dense.activation.apply(acc);
        }
        ops.add_matmul(dense.output_size, 1, dense.input_size);
        ops.add_memory_read(w_len * core::mem::size_of::<f32>());
        ops.add_activation(dense.output_size);
        ops.add_memory_write(dense.output_size * core::mem::size_of::<f32>());
    }
    let out_idx = topo.len() - 1;
    Some((offs[out_idx], topo[out_idx]))
}

fn eval_forward_f64(
    topo: &[usize],
    weights: &[f64],
    biases: &[f64],
    layer_specs: &[LayerSpec],
    input: &[f64],
    activations: &mut [f64],
    ops: &mut crate::observability::profiler::OpCounter,
) -> Option<(usize, usize)> {
    let mut offs = [0usize; MAX_LAYERS + 1];
    if topo.len() >= offs.len() {
        return None;
    }
    let mut running = 0usize;
    for (i, &size) in topo.iter().enumerate() {
        offs[i] = running;
        running = running.checked_add(size)?;
    }
    if activations.len() < running || input.len() != topo[0] {
        return None;
    }
    activations[..topo[0]].copy_from_slice(input);
    for (layer_idx, spec) in layer_specs.iter().enumerate() {
        let LayerSpec::Dense(dense) = *spec;
        let prev_off = offs[layer_idx];
        let curr_off = offs[layer_idx + 1];
        let (left, right) = activations.split_at_mut(curr_off);
        let prev = &left[prev_off..prev_off + dense.input_size];
        let curr = &mut right[..dense.output_size];
        let w_len = dense.input_size.checked_mul(dense.output_size)?;
        let w = &weights[dense.weight_offset..dense.weight_offset + w_len];
        let b = &biases[dense.bias_offset..dense.bias_offset + dense.output_size];
        for o in 0..dense.output_size {
            let row = o * dense.input_size;
            let mut acc = b[o];
            for i in 0..dense.input_size {
                acc += w[row + i] * prev[i];
            }
            curr[o] = dense.activation.apply_f64(acc);
        }
        ops.add_matmul(dense.output_size, 1, dense.input_size);
        ops.add_memory_read(w_len * core::mem::size_of::<f64>());
        ops.add_activation(dense.output_size);
        ops.add_memory_write(dense.output_size * core::mem::size_of::<f64>());
    }
    let out_idx = topo.len() - 1;
    Some((offs[out_idx], topo[out_idx]))
}

pub(crate) fn evaluate_on_decrypted(
    bytes: &mut [u8],
    samples: &[(&[f32], &[f32])],
    layout: &ParsedLayout,
    accum: &mut EvalAccum,
) -> Result<(), RnnFlowError> {
    let topo = &layout.topology[..layout.topo_len];
    let topo_len = layout.topo_len;
    let dtype = layout.dtype;
    let w_start = layout.w_start;
    let w_end = layout.w_end;
    let b_start = layout.b_start;
    let b_end = layout.b_end;
    let hidden_activation = layout.hidden_activation;
    let output_activation = layout.output_activation;
    let buf_len = match required_train_buffer_len(topo) {
        Some(n) if n <= MAX_TOTAL_NEURONS => n,
        _ => return Err(RnnFlowError::BadBytes),
    };
    let lc = topo_len.saturating_sub(1);
    match dtype {
        0 => {
            if !(w_end - w_start).is_multiple_of(4) || !(b_end - b_start).is_multiple_of(4) {
                return Err(RnnFlowError::BadBytes);
            }
            let mut layer_specs_buf = [LayerSpec::Dense(crate::graph::net::layers::LayerDesc {
                input_size: 1, output_size: 1, weight_offset: 0, bias_offset: 0,
                activation: ActivationKind::Identity,
            }); MAX_LAYERS];
            let mut activations_buf = [0.0f32; MAX_TOTAL_NEURONS];
            let mut probs_buf = [0.0f32; MAX_TOTAL_NEURONS];
            let (weights, biases) = if w_start < b_start {
                if w_end > b_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(b_start);
                (bytes_as_f32_mut(&mut left[w_start..w_end]), bytes_as_f32_mut(&mut right[..b_end - b_start]))
            } else {
                if b_end > w_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(w_start);
                (bytes_as_f32_mut(&mut right[..w_end - w_start]), bytes_as_f32_mut(&mut left[b_start..b_end]))
            };
            let layer_count = build_from_layers(
                topo, hidden_activation, output_activation,
                weights.len(), biases.len(), &mut layer_specs_buf[..lc],
            ).map_err(|_| RnnFlowError::BadBytes)?;
            let specs = &layer_specs_buf[..layer_count];
            for &(input, target) in samples {
                if input.len() != topo[0] || target.len() != topo[topo_len - 1] { continue; }
                let acts = &mut activations_buf[..buf_len];
                let Some((out_off, out_size)) = eval_forward_f32(topo, weights, biases, specs, input, acts, &mut accum.ops) else { continue; };
                if target.len() != out_size { continue; }
                let out = &acts[out_off..out_off + out_size];
                let probs = &mut probs_buf[..out_size];
                if crate::engine::infer::inference::softmax_stable(out, probs).is_err() { continue; }
                if let Ok(v) = crate::engine::eval::metrics::mse_f32(out, target) { accum.mse.update(v); }
                if let Ok(v) = crate::engine::eval::metrics::mae_f32(out, target) { accum.mae.update(v); }
                if let Ok(v) = crate::engine::eval::metrics::accuracy_top1_from_one_hot_f32(out, target) { accum.accuracy.update(v); }
                if let Ok(v) = crate::engine::eval::metrics::cross_entropy_from_probabilities_f32(probs, target, f32::EPSILON) { accum.cross_entropy.update(v); }
                if let Some(idx) = crate::engine::eval::metrics::argmax_f32(out) { accum.confidence.update(probs[idx]); }
            }
        }
        1 => {
            if !(w_end - w_start).is_multiple_of(8) || !(b_end - b_start).is_multiple_of(8) {
                return Err(RnnFlowError::BadBytes);
            }
            let mut layer_specs_buf = [LayerSpec::Dense(crate::graph::net::layers::LayerDesc {
                input_size: 1, output_size: 1, weight_offset: 0, bias_offset: 0,
                activation: ActivationKind::Identity,
            }); MAX_LAYERS];
            let mut activations_buf = [0.0f64; MAX_TOTAL_NEURONS];
            let mut probs_buf = [0.0f64; MAX_TOTAL_NEURONS];
            let mut input_f64 = [0.0f64; MAX_TOTAL_NEURONS];
            let mut target_f64 = [0.0f64; MAX_TOTAL_NEURONS];
            let (weights, biases) = if w_start < b_start {
                if w_end > b_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(b_start);
                (bytes_as_f64_mut(&mut left[w_start..w_end]), bytes_as_f64_mut(&mut right[..b_end - b_start]))
            } else {
                if b_end > w_start { return Err(RnnFlowError::BadBytes); }
                let (left, right) = bytes.split_at_mut(w_start);
                (bytes_as_f64_mut(&mut right[..w_end - w_start]), bytes_as_f64_mut(&mut left[b_start..b_end]))
            };
            let layer_count = build_from_layers(
                topo, hidden_activation, output_activation,
                weights.len(), biases.len(), &mut layer_specs_buf[..lc],
            ).map_err(|_| RnnFlowError::BadBytes)?;
            let specs = &layer_specs_buf[..layer_count];
            for &(input, target) in samples {
                if input.len() != topo[0] || target.len() != topo[topo_len - 1] { continue; }
                for (i, &v) in input.iter().enumerate() { input_f64[i] = v as f64; }
                for (i, &v) in target.iter().enumerate() { target_f64[i] = v as f64; }
                let acts = &mut activations_buf[..buf_len];
                let Some((out_off, out_size)) = eval_forward_f64(topo, weights, biases, specs, &input_f64[..input.len()], acts, &mut accum.ops) else { continue; };
                if target.len() != out_size { continue; }
                let out = &acts[out_off..out_off + out_size];
                let target_slice = &target_f64[..out_size];
                let probs = &mut probs_buf[..out_size];
                if crate::engine::infer::inference::softmax_stable(out, probs).is_err() { continue; }
                if let Ok(v) = crate::engine::eval::metrics::mse_f64(out, target_slice) { accum.mse.update(v as f32); }
                if let Ok(v) = crate::engine::eval::metrics::mae_f64(out, target_slice) { accum.mae.update(v as f32); }
                if let Ok(v) = crate::engine::eval::metrics::accuracy_top1_from_one_hot_f64(out, target_slice) { accum.accuracy.update(v as f32); }
                if let Ok(v) = crate::engine::eval::metrics::cross_entropy_from_probabilities_f64(probs, target_slice, f64::EPSILON) { accum.cross_entropy.update(v as f32); }
                if let Some(idx) = crate::engine::eval::metrics::argmax_f64(out) { accum.confidence.update(probs[idx] as f32); }
            }
        }
        _ => return Err(RnnFlowError::BadBytes),
    }
    Ok(())
}
