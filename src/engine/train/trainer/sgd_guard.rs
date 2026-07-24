use crate::engine::train::trainer::SgdConfig;

pub fn config_is_valid(config: SgdConfig) -> bool {
    config.learning_rate.is_finite() && config.learning_rate > 0.0
}

pub fn io_matches_layers(layers: &[usize], input_len: usize, target_len: usize) -> bool {
    if layers.len() < 2 {
        return false;
    }
    input_len == layers[0] && target_len == layers[layers.len() - 1]
}
