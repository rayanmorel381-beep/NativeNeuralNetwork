use crate::graph::blocks::embeddings::{gather_embeddings, mlp_head_forward};
use crate::graph::attn::kv_cache::KvCacheView;
use crate::graph::blocks::normalization::rms_norm_in_place;
use crate::graph::lm::{LmConfig, LmError, LmWeights};
use crate::graph::lm::{
    transformer_block_forward_lora, AdaptedWeightBuf, LmBlockLora, LoraForward,
};
use crate::base::math::Float;
use super::activations::TrainActivations;
use super::scratch::ForwardScratch;
use super::transformer_block::{transformer_block_forward, BlockQuant};

pub fn transformer_forward<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    token_id: u32,
    position: usize,
    kv_layers: &mut [KvCacheView<T>],
    sc: &mut ForwardScratch<T>,
) -> Result<(), LmError> {
    transformer_forward_inner(cfg, w, token_id, position, kv_layers, sc, None)
}

pub struct LoraWeights<'a, T: Float> {
    pub loras: &'a [LmBlockLora<'a, T>],
    pub adapted: &'a mut AdaptedWeightBuf<'a, T>,
}

pub fn transformer_forward_lora<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    lw: &mut LoraWeights<'_, T>,
    token_id: u32,
    position: usize,
    kv_layers: &mut [KvCacheView<T>],
    sc: &mut ForwardScratch<T>,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;

    if kv_layers.len() < cfg.num_layers || lw.loras.len() < cfg.num_layers {
        return Err(LmError::ShapeMismatch);
    }
    if sc.hidden.len() < h || sc.logits.len() < v {
        return Err(LmError::ShapeMismatch);
    }

    gather_embeddings(w.embed_table, v, h, &[token_id as usize], sc.hidden)
        .map_err(|_| LmError::ShapeMismatch)?;

    for (layer, (block, kv)) in w.blocks.iter().zip(kv_layers.iter_mut()).enumerate() {
        transformer_block_forward_lora(
            LoraForward {
                cfg,
                base_w: block,
                lora: &lw.loras[layer],
            },
            AdaptedWeightBuf {
                wq: &mut *lw.adapted.wq,
                wk: &mut *lw.adapted.wk,
                wv: &mut *lw.adapted.wv,
                wo: &mut *lw.adapted.wo,
            },
            position,
            kv,
            sc,
            None,
        )?;
    }

    rms_norm_in_place(&mut sc.hidden[..h], w.final_norm_gamma, T::from_f32(1e-5))
        .map_err(|_| LmError::ShapeMismatch)?;

    let hh = cfg.head_hidden;
    mlp_head_forward(
        &sc.hidden[..h],
        w.head.w1,
        w.head.b1,
        w.head.w2,
        w.head.b2,
        v,
        h,
        hh,
        sc.head_z1,
        sc.head_a1,
        sc.logits,
    )
    .map_err(|_| LmError::ShapeMismatch)?;

    Ok(())
}

/// Forward pass that also fills per-layer activations for BPTT.
pub fn transformer_forward_train<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    token_id: u32,
    position: usize,
    kv_layers: &mut [KvCacheView<T>],
    sc: &mut ForwardScratch<T>,
    train_acts: &mut TrainActivations<T>,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;

    if kv_layers.len() < cfg.num_layers || train_acts.blocks.len() < cfg.num_layers {
        return Err(LmError::ShapeMismatch);
    }
    if sc.hidden.len() < h || sc.logits.len() < v {
        return Err(LmError::ShapeMismatch);
    }

    gather_embeddings(w.embed_table, v, h, &[token_id as usize], sc.hidden)
        .map_err(|_| LmError::ShapeMismatch)?;

    for layer in 0..cfg.num_layers {
        let (left, right) = train_acts.blocks.split_at_mut(layer + 1);
        let _ = right;
        let block_acts = &mut left[layer];
        transformer_block_forward(
            cfg,
            &w.blocks[layer],
            position,
            &mut kv_layers[layer],
            sc,
            BlockQuant { qw: None, tw: None },
            Some(block_acts),
            None,
        )?;
    }

    if train_acts.pre_final_hidden.len() >= h {
        train_acts.pre_final_hidden[..h].copy_from_slice(&sc.hidden[..h]);
    }

    rms_norm_in_place(&mut sc.hidden[..h], w.final_norm_gamma, T::from_f32(1e-5))
        .map_err(|_| LmError::ShapeMismatch)?;

    if train_acts.final_hidden.len() >= h {
        train_acts.final_hidden[..h].copy_from_slice(&sc.hidden[..h]);
    }

    let hh = cfg.head_hidden;
    mlp_head_forward(
        &sc.hidden[..h],
        w.head.w1,
        w.head.b1,
        w.head.w2,
        w.head.b2,
        v,
        h,
        hh,
        sc.head_z1,
        sc.head_a1,
        sc.logits,
    )
    .map_err(|_| LmError::ShapeMismatch)?;

    Ok(())
}

fn transformer_forward_inner<T: Float>(
    cfg: &LmConfig<T>,
    w: &LmWeights<T>,
    token_id: u32,
    position: usize,
    kv_layers: &mut [KvCacheView<T>],
    sc: &mut ForwardScratch<T>,
    _train_acts: Option<&mut TrainActivations<T>>,
) -> Result<(), LmError> {
    let ctx = crate::graph::propagation::TransformCtx::lm(cfg, w, token_id, position);
    if kv_layers.len() < cfg.num_layers || w.blocks.len() < cfg.num_layers {
        return Err(LmError::ShapeMismatch);
    }
    if sc.hidden.len() < cfg.hidden_size || sc.logits.len() < cfg.vocab_size {
        return Err(LmError::ShapeMismatch);
    }
    crate::graph::scheduler::run_lmlp(
        &ctx,
        crate::graph::topology::LmlpTopology::Lm {
            num_layers: cfg.num_layers,
        },
        kv_layers,
        sc,
    )
    .map(|_| ())
}
