use crate::graph::lm::LmConfig;

pub struct BlockActivations<'a, T> {
    pub pre_norm_attn: &'a mut [T],
    pub post_norm_attn: &'a mut [T],
    pub q: &'a mut [T],
    pub k_proj: &'a mut [T],
    pub v_proj: &'a mut [T],
    pub attn_scores_per_head: &'a mut [T],
    pub pre_wo: &'a mut [T],
    pub pre_norm_ffn: &'a mut [T],
    pub post_norm_ffn: &'a mut [T],
    pub gate_pre: &'a mut [T],
    pub up: &'a mut [T],
    pub seq_len: usize,
}

impl<'a, T: crate::base::math::Float> BlockActivations<'a, T> {
    pub fn f32_count(cfg: &LmConfig<T>) -> usize {
        let h = cfg.hidden_size;
        let f = cfg.ffw_size;
        let nh = cfg.num_heads;
        let kv_h = cfg.kv_h();
        let ctx = cfg.context_len;
        2 * h + h + 2 * kv_h + nh * ctx + h + 2 * h + 2 * f
    }
}

pub struct TrainActivations<'a, T> {
    pub blocks: &'a mut [BlockActivations<'a, T>],
    pub pre_final_hidden: &'a mut [T],
    pub final_hidden: &'a mut [T],
    pub probs: &'a mut [T],
}
