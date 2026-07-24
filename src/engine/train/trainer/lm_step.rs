use crate::graph::attn::kv_cache::KvCacheView;
use crate::engine::eval::losses::{cross_entropy, cross_entropy_grad};
use crate::engine::train::optimizers::{step_adamw, AdamwConfig};
use crate::engine::train::gradients::clip_by_global_norm;
use crate::engine::train::schedulers::{compute_learning_rate, LrSchedule};
use crate::engine::infer::inference::{BackwardScratch, mlp_head_backward, transformer_block_backward,
    TrainActivations, ForwardScratch};
use crate::engine::infer::inference::transformer_forward::transformer_forward_train;
use crate::graph::lm::sub_agents::LmConfig;
use crate::graph::lm::errors::LmError;
use crate::graph::lm::grads::LmWeightsGrads;
use crate::graph::lm::weights::LmWeights;
use crate::engine::train::optimizers::OptimizerError;
use crate::base::math::Float;

#[derive(Clone, Copy)]
pub struct LmTrainConfig<T: Float> {
    pub base_lr: T,
    pub schedule: LrSchedule<T>,
    pub grad_clip: T,
    pub weight_decay: T,
    pub beta1: T,
    pub beta2: T,
    pub pad_id: u32,
}

impl<T: Float> LmTrainConfig<T> {
    pub fn default_adam() -> Self {
        Self {
            base_lr: T::from_f32(3e-4),
            schedule: LrSchedule::Constant,
            grad_clip: T::ONE,
            weight_decay: T::from_f32(0.01),
            beta1: T::from_f32(0.9),
            beta2: T::from_f32(0.999),
            pad_id: 0,
        }
    }

    pub fn with_warmup(base_lr: T, warmup_steps: u32) -> Self {
        Self {
            base_lr,
            schedule: LrSchedule::LinearWarmup { warmup_steps },
            grad_clip: T::ONE,
            weight_decay: T::from_f32(0.01),
            beta1: T::from_f32(0.9),
            beta2: T::from_f32(0.999),
            pad_id: 0,
        }
    }
}

pub struct LmTrainStep<T> {
    pub loss: T,
    pub grad_norm: T,
    pub lr: T,
    pub ops: crate::observability::profiler::OpCounter,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LmTrainMode {
    Batched,
    Sequential,
    SequentialMasked,
    Accumulate,
    AccumulateMasked,
    MultiSequence,
}

/// Full training step: forward → backward → clip → AdamW.
///
/// `token_ids` is a sequence of at least 2 tokens.  Each consecutive pair
/// (input, target) forms one training sample in the mini-batch.
///
/// The caller manages all memory: `w_params`, `w_grads`, `w_m`, `w_v` must
/// match `LmWeights::param_count(cfg)` exactly.
///
/// `train_acts_buf` must be sized to hold activations for all layers:
/// `BlockActivations::f32_count(cfg) * cfg.num_layers + 2 * cfg.hidden_size + cfg.vocab_size`.
/// The caller also provides `back_sc` for backward scratch.
pub struct AdamState<'a, T> {
    pub params: &'a mut [T],
    pub grads_buf: &'a mut [T],
    pub m: &'a mut [T],
    pub v: &'a mut [T],
    pub step: u32,
}

pub struct TrainScratch<'a, T> {
    pub kv_layers: &'a mut [KvCacheView<'a, T>],
    pub sc: &'a mut ForwardScratch<'a, T>,
    pub train_acts: &'a mut TrainActivations<'a, T>,
    pub back_sc: &'a mut BackwardScratch<'a, T>,
}

pub fn train_step<T: Float>(
    cfg: &LmConfig<T>,
    train_cfg: &LmTrainConfig<T>,
    adam: AdamState<'_, T>,
    w: &LmWeights<T>,
    grads: &mut LmWeightsGrads<T>,
    token_ids: &[u32],
    scratch: TrainScratch<'_, T>,
) -> Result<LmTrainStep<T>, LmError> {
    train_step_inner(cfg, train_cfg, adam, w, grads, token_ids, None, scratch, true)
}

#[allow(clippy::too_many_arguments)]
pub fn train_step_masked<T: Float>(
    cfg: &LmConfig<T>,
    train_cfg: &LmTrainConfig<T>,
    adam: AdamState<'_, T>,
    w: &LmWeights<T>,
    grads: &mut LmWeightsGrads<T>,
    token_ids: &[u32],
    target_active: &[bool],
    scratch: TrainScratch<'_, T>,
) -> Result<LmTrainStep<T>, LmError> {
    if target_active.len() + 1 != token_ids.len() {
        return Err(LmError::ShapeMismatch);
    }
    train_step_inner(cfg, train_cfg, adam, w, grads, token_ids, Some(target_active), scratch, true)
}

pub fn accumulate_grads<T: Float>(
    cfg: &LmConfig<T>,
    train_cfg: &LmTrainConfig<T>,
    adam: AdamState<'_, T>,
    w: &LmWeights<T>,
    grads: &mut LmWeightsGrads<T>,
    token_ids: &[u32],
    scratch: TrainScratch<'_, T>,
) -> Result<LmTrainStep<T>, LmError> {
    train_step_inner(cfg, train_cfg, adam, w, grads, token_ids, None, scratch, false)
}

#[allow(clippy::too_many_arguments)]
pub fn accumulate_grads_masked<T: Float>(
    cfg: &LmConfig<T>,
    train_cfg: &LmTrainConfig<T>,
    adam: AdamState<'_, T>,
    w: &LmWeights<T>,
    grads: &mut LmWeightsGrads<T>,
    token_ids: &[u32],
    target_active: &[bool],
    scratch: TrainScratch<'_, T>,
) -> Result<LmTrainStep<T>, LmError> {
    if target_active.len() + 1 != token_ids.len() {
        return Err(LmError::ShapeMismatch);
    }
    train_step_inner(cfg, train_cfg, adam, w, grads, token_ids, Some(target_active), scratch, false)
}


pub fn apply_grads_adamw<T: Float>(
    train_cfg: &LmTrainConfig<T>,
    w_params: &mut [T],
    w_grads: &mut [T],
    w_m: &mut [T],
    w_v: &mut [T],
    step: u32,
) -> Result<T, LmError> {
    if w_grads.len() != w_params.len()
        || w_m.len() != w_params.len()
        || w_v.len() != w_params.len()
    {
        return Err(LmError::ShapeMismatch);
    }
    let grad_norm = if train_cfg.grad_clip > T::ZERO {
        clip_by_global_norm(w_grads, train_cfg.grad_clip).unwrap_or(T::ZERO)
    } else {
        T::ZERO
    };
    let lr = compute_learning_rate(train_cfg.base_lr, step, train_cfg.schedule)
        .unwrap_or(train_cfg.base_lr);
    step_adamw(
        w_params,
        w_grads,
        w_m,
        w_v,
        AdamwConfig {
            learning_rate: lr,
            step,
            beta1: train_cfg.beta1,
            beta2: train_cfg.beta2,
            eps: T::from_f32(1e-8),
            weight_decay: train_cfg.weight_decay,
        },
    )
    .map_err(|e| match e {
        OptimizerError::StepOverflow => LmError::InvalidConfig,
        OptimizerError::InvalidHyperParams => LmError::InvalidConfig,
        OptimizerError::ShapeMismatch => LmError::ShapeMismatch,
    })?;
    for g in w_grads.iter_mut() {
        *g = T::ZERO;
    }
    Ok(grad_norm)
}

#[allow(clippy::too_many_arguments)]
fn run_lm_backward<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    grads: &mut LmWeightsGrads<T>,
    back_sc: &mut BackwardScratch<'_, T>,
    kv_layers: &[KvCacheView<'_, T>],
    sc: &mut ForwardScratch<'_, T>,
    train_acts: &TrainActivations<'_, T>,
    input_tok: u32,
    target_tok: usize,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let hh = cfg.head_hidden;

    cross_entropy_grad(&train_acts.probs[..v], target_tok, &mut sc.logits[..v])
        .map_err(|_| LmError::NonFinite)?;

    mlp_head_backward(
        &sc.logits[..v],
        &train_acts.final_hidden[..h],
        &sc.head_a1[..hh],
        w.head.w1,
        w.head.w2,
        v,
        h,
        hh,
        grads.head_w1,
        grads.head_b1,
        grads.head_w2,
        grads.head_b2,
        &mut sc.head_z1[..hh],
        &mut sc.hidden[..h],
    )?;

    let d_out = &mut sc.wo_out[..h];
    for x in d_out.iter_mut() {
        *x = T::ZERO;
    }
    crate::engine::infer::inference::rmsnorm_backward(
        &train_acts.pre_final_hidden[..h],
        w.final_norm_gamma,
        &sc.hidden[..h],
        &mut grads.final_norm_gamma[..h],
        d_out,
        h,
    );

    for layer in (0..cfg.num_layers).rev() {
        transformer_block_backward(
            cfg,
            &w.blocks[layer],
            &train_acts.blocks[layer],
            &kv_layers[layer],
            d_out,
            &mut grads.blocks[layer],
            back_sc,
        )?;
    }

    let embed_off = input_tok as usize * h;
    if embed_off + h <= grads.embed_table.len() {
        let dst = &mut grads.embed_table[embed_off..embed_off + h];
        for (d, &g) in dst.iter_mut().zip(d_out.iter().take(h)) {
            *d += g;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn train_step_inner<T: Float>(
    cfg: &LmConfig<T>,
    train_cfg: &LmTrainConfig<T>,
    adam: AdamState<'_, T>,
    w: &LmWeights<T>,
    grads: &mut LmWeightsGrads<T>,
    token_ids: &[u32],
    target_active: Option<&[bool]>,
    scratch: TrainScratch<'_, T>,
    apply_step: bool,
) -> Result<LmTrainStep<T>, LmError> {
    let AdamState { params: w_params, grads_buf: w_grads, m: w_m, v: w_v, step } = adam;
    let TrainScratch { kv_layers, sc, train_acts, back_sc } = scratch;
    if token_ids.len() < 2 {
        return Err(LmError::ShapeMismatch);
    }
    if w_grads.len() != w_params.len() {
        return Err(LmError::ShapeMismatch);
    }
    if apply_step && (w_m.len() != w_params.len() || w_v.len() != w_params.len()) {
        return Err(LmError::ShapeMismatch);
    }
    if train_acts.blocks.len() < cfg.num_layers {
        return Err(LmError::ShapeMismatch);
    }
    if train_acts.probs.len() < cfg.vocab_size {
        return Err(LmError::ScratchTooSmall);
    }

    grads.zero();
    let mut total_loss = T::ZERO;
    let n_steps = token_ids.len() - 1;
    let mut active_count = 0usize;

    let lr = compute_learning_rate(train_cfg.base_lr, step, train_cfg.schedule)
        .unwrap_or(train_cfg.base_lr);

    for t in 0..n_steps {
        let input_tok = token_ids[t];
        let target_tok = token_ids[t + 1] as usize;
        let is_active = match target_active {
            Some(mask) => mask[t],
            None => true,
        };

        sc.moe_aux_loss = 0.0;
        transformer_forward_train(cfg, w, input_tok, t, kv_layers, sc, train_acts)?;

        if !is_active {
            continue;
        }

        let loss = cross_entropy(
            &sc.logits[..cfg.vocab_size],
            target_tok,
            train_acts.probs,
        )
        .map_err(|_| LmError::NonFinite)?
            + T::from_f32(sc.moe_aux_loss) * T::from_f32(0.01);
        total_loss += loss;
        active_count += 1;

        if target_tok >= train_acts.probs.len() {
            return Err(LmError::NonFinite);
        }

        run_lm_backward(
            cfg, w, grads, back_sc, kv_layers, sc, train_acts, input_tok, target_tok,
        )?;
    }

    let denom = if target_active.is_some() {
        active_count
    } else {
        n_steps
    };
    if denom == 0 {
        return Ok(LmTrainStep { loss: T::ZERO, grad_norm: T::ZERO, lr, ops: crate::observability::profiler::OpCounter::new() });
    }
    let inv_n = T::ONE / T::from_usize(denom);
    total_loss *= inv_n;

    // Scale gradients by 1/denom.
    for g in w_grads.iter_mut() {
        *g *= inv_n;
    }

    if !apply_step {
        return Ok(LmTrainStep { loss: total_loss, grad_norm: T::ZERO, lr, ops: crate::observability::profiler::OpCounter::new() });
    }

    // Gradient clipping.
    let grad_norm = if train_cfg.grad_clip > T::ZERO {
        clip_by_global_norm(w_grads, train_cfg.grad_clip)
            .unwrap_or(T::ZERO)
    } else {
        T::ZERO
    };

    // AdamW optimizer step.
    step_adamw(
        w_params,
        w_grads,
        w_m,
        w_v,
        AdamwConfig {
            learning_rate: lr,
            step,
            beta1: train_cfg.beta1,
            beta2: train_cfg.beta2,
            eps: T::from_f32(1e-8),
            weight_decay: train_cfg.weight_decay,
        },
    )
    .map_err(|e| match e {
        OptimizerError::StepOverflow => LmError::InvalidConfig,
        OptimizerError::InvalidHyperParams => LmError::InvalidConfig,
        OptimizerError::ShapeMismatch => LmError::ShapeMismatch,
    })?;

    for g in w_grads.iter_mut() {
        *g = T::ZERO;
    }

    Ok(LmTrainStep { loss: total_loss, grad_norm, lr, ops: crate::observability::profiler::OpCounter::new() })
}

