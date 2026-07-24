use crate::graph::lm::LmConfig;

pub struct ForwardScratch<'a, T> {
    pub hidden:        &'a mut [T],
    pub norm_buf:      &'a mut [T],
    pub qkv:           &'a mut [T],
    pub wo_out:        &'a mut [T],
    pub scores:        &'a mut [T],
    pub attn_head:     &'a mut [T],
    pub ffn_gate:      &'a mut [T],
    pub ffn_up:        &'a mut [T],
    pub ffn_out:       &'a mut [T],
    pub logits:        &'a mut [T],
    pub head_z1:       &'a mut [T],
    pub head_a1:       &'a mut [T],
    pub moe_aux_loss:  f32,
}

impl<'a, T: crate::base::math::Float> ForwardScratch<'a, T> {
    pub fn total_f32_count(cfg: &LmConfig<T>) -> usize {
        let h = cfg.hidden_size;
        let f = cfg.ffw_size;
        let c = cfg.context_len;
        let v = cfg.vocab_size;
        let hh = cfg.head_hidden;
        let kv_h = cfg.kv_h();
        h + h + (h + 2 * kv_h) + h + c + h + f + f + h + v + hh + hh
    }

    pub fn validate_sizes(&self, cfg: &LmConfig<T>) -> bool {
        let h = cfg.hidden_size;
        let f = cfg.ffw_size;
        let c = cfg.context_len;
        let v = cfg.vocab_size;
        let hh = cfg.head_hidden;
        let kv_h = cfg.kv_h();
        self.hidden.len() >= h
            && self.norm_buf.len() >= h
            && self.qkv.len() >= h + 2 * kv_h
            && self.wo_out.len() >= h
            && self.scores.len() >= c
            && self.attn_head.len() >= h
            && self.ffn_gate.len() >= f
            && self.ffn_up.len() >= f
            && self.ffn_out.len() >= h
            && self.logits.len() >= v
            && self.head_z1.len() >= hh
            && self.head_a1.len() >= hh
    }
}

