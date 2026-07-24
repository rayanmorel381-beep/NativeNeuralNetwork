use crate::base::math::{sqrtd, sqrtf};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitKindF32 {
    Zeros,
    XavierUniform,
    HeUniform,
    Constant(f32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InitKindF64 {
    Zeros,
    XavierUniform,
    HeUniform,
    Constant(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InitError {
    InvalidShape,
    ShapeMismatch,
    NonFinite,
}

pub fn expected_parameter_counts(layers: &[usize]) -> Option<(usize, usize)> {
    if layers.len() < 2 {
        return None;
    }

    let mut weights = 0usize;
    let mut biases = 0usize;
    for i in 0..layers.len() - 1 {
        let in_size = layers[i];
        let out_size = layers[i + 1];
        if in_size == 0 || out_size == 0 {
            return None;
        }
        weights = weights.checked_add(in_size.checked_mul(out_size)?)?;
        biases = biases.checked_add(out_size)?;
    }
    Some((weights, biases))
}

fn initialize_parameters_f32_impl(
    layers: &[usize],
    weights: &mut [f32],
    biases: &mut [f32],
    kind: InitKindF32,
    rng: &mut SplitMix64,
) -> Result<(), InitError> {
    let (expected_w, expected_b) =
        expected_parameter_counts(layers).ok_or(InitError::InvalidShape)?;
    if weights.len() != expected_w || biases.len() != expected_b {
        return Err(InitError::ShapeMismatch);
    }

    let mut w_off = 0usize;
    let mut b_off = 0usize;
    let six: f32 = 6.0;

    for i in 0..layers.len() - 1 {
        let in_size = layers[i];
        let out_size = layers[i + 1];
        let w_len = in_size * out_size;
        let (w_slice, b_slice) = (
            &mut weights[w_off..w_off + w_len],
            &mut biases[b_off..b_off + out_size],
        );

        match kind {
            InitKindF32::Zeros => {
                for w in w_slice {
                    *w = 0.0;
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
            InitKindF32::Constant(value) => {
                if !value.is_finite() {
                    return Err(InitError::NonFinite);
                }
                for w in w_slice {
                    *w = value;
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
            InitKindF32::XavierUniform => {
                let denom = (in_size + out_size) as f32;
                if denom <= 0.0 || !denom.is_finite() {
                    return Err(InitError::NonFinite);
                }
                let limit = sqrtf(six / denom);
                if !limit.is_finite() {
                    return Err(InitError::NonFinite);
                }
                for w in w_slice {
                    *w = random_uniform_symmetric_f32(rng, limit);
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
            InitKindF32::HeUniform => {
                let denom = in_size as f32;
                if denom <= 0.0 || !denom.is_finite() {
                    return Err(InitError::NonFinite);
                }
                let limit = sqrtf(six / denom);
                if !limit.is_finite() {
                    return Err(InitError::NonFinite);
                }
                for w in w_slice {
                    *w = random_uniform_symmetric_f32(rng, limit);
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
        }

        w_off += w_len;
        b_off += out_size;
    }

    Ok(())
}

fn initialize_parameters_f64_impl(
    layers: &[usize],
    weights: &mut [f64],
    biases: &mut [f64],
    kind: InitKindF64,
    rng: &mut SplitMix64,
) -> Result<(), InitError> {
    let (expected_w, expected_b) =
        expected_parameter_counts(layers).ok_or(InitError::InvalidShape)?;
    if weights.len() != expected_w || biases.len() != expected_b {
        return Err(InitError::ShapeMismatch);
    }

    let mut w_off = 0usize;
    let mut b_off = 0usize;
    let six: f64 = 6.0;

    for i in 0..layers.len() - 1 {
        let in_size = layers[i];
        let out_size = layers[i + 1];
        let w_len = in_size * out_size;
        let (w_slice, b_slice) = (
            &mut weights[w_off..w_off + w_len],
            &mut biases[b_off..b_off + out_size],
        );

        match kind {
            InitKindF64::Zeros => {
                for w in w_slice {
                    *w = 0.0;
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
            InitKindF64::Constant(value) => {
                if !value.is_finite() {
                    return Err(InitError::NonFinite);
                }
                for w in w_slice {
                    *w = value;
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
            InitKindF64::XavierUniform => {
                let denom = (in_size + out_size) as f64;
                if denom <= 0.0 || !denom.is_finite() {
                    return Err(InitError::NonFinite);
                }
                let limit = sqrtd(six / denom);
                if !limit.is_finite() {
                    return Err(InitError::NonFinite);
                }
                for w in w_slice {
                    *w = random_uniform_symmetric_f64(rng, limit);
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
            InitKindF64::HeUniform => {
                let denom = in_size as f64;
                if denom <= 0.0 || !denom.is_finite() {
                    return Err(InitError::NonFinite);
                }
                let limit = sqrtd(six / denom);
                if !limit.is_finite() {
                    return Err(InitError::NonFinite);
                }
                for w in w_slice {
                    *w = random_uniform_symmetric_f64(rng, limit);
                }
                for b in b_slice {
                    *b = 0.0;
                }
            }
        }

        w_off += w_len;
        b_off += out_size;
    }

    Ok(())
}

pub fn initialize_parameters_f32(
    layers: &[usize],
    weights: &mut [f32],
    biases: &mut [f32],
    init_kind: InitKindF32,
    seed: u64,
) -> Result<(), InitError> {
    match init_kind {
        InitKindF32::Zeros => initialize_parameters_f32_zeros(layers, weights, biases),
        InitKindF32::XavierUniform => {
            let mut rng = SplitMix64::new(seed);
            initialize_parameters_f32_impl(layers, weights, biases, InitKindF32::XavierUniform, &mut rng)
        }
        InitKindF32::HeUniform => initialize_parameters_f32_he_uniform(layers, weights, biases, seed),
        InitKindF32::Constant(value) => initialize_parameters_f32_constant(layers, weights, biases, value),
    }
}

pub fn initialize_parameters_f64(
    layers: &[usize],
    weights: &mut [f64],
    biases: &mut [f64],
    init_kind: InitKindF64,
    seed: u64,
) -> Result<(), InitError> {
    match init_kind {
        InitKindF64::Zeros => initialize_parameters_f64_zeros(layers, weights, biases),
        InitKindF64::XavierUniform => {
            let mut rng = SplitMix64::new(seed.wrapping_add(0xC6A4A7935BD1E995));
            initialize_parameters_f64_impl(layers, weights, biases, InitKindF64::XavierUniform, &mut rng)
        }
        InitKindF64::HeUniform => initialize_parameters_f64_he_uniform(layers, weights, biases, seed),
        InitKindF64::Constant(value) => initialize_parameters_f64_constant(layers, weights, biases, value),
    }
}

pub fn initialize_parameters_f32_zeros(
    layers: &[usize],
    weights: &mut [f32],
    biases: &mut [f32],
) -> Result<(), InitError> {
    let mut rng = SplitMix64::new(0);
    initialize_parameters_f32_impl(layers, weights, biases, InitKindF32::Zeros, &mut rng)
}

pub fn initialize_parameters_f64_zeros(
    layers: &[usize],
    weights: &mut [f64],
    biases: &mut [f64],
) -> Result<(), InitError> {
    let mut rng = SplitMix64::new(0);
    initialize_parameters_f64_impl(layers, weights, biases, InitKindF64::Zeros, &mut rng)
}

pub fn initialize_parameters_f32_he_uniform(
    layers: &[usize],
    weights: &mut [f32],
    biases: &mut [f32],
    seed: u64,
) -> Result<(), InitError> {
    let mut rng = SplitMix64::new(seed);
    initialize_parameters_f32_impl(layers, weights, biases, InitKindF32::HeUniform, &mut rng)
}

pub fn initialize_parameters_f64_he_uniform(
    layers: &[usize],
    weights: &mut [f64],
    biases: &mut [f64],
    seed: u64,
) -> Result<(), InitError> {
    let mut rng = SplitMix64::new(seed.wrapping_add(0xC6A4A7935BD1E995));
    initialize_parameters_f64_impl(layers, weights, biases, InitKindF64::HeUniform, &mut rng)
}

pub fn initialize_parameters_f32_constant(
    layers: &[usize],
    weights: &mut [f32],
    biases: &mut [f32],
    value: f32,
) -> Result<(), InitError> {
    let mut rng = SplitMix64::new(0);
    initialize_parameters_f32_impl(layers, weights, biases, InitKindF32::Constant(value), &mut rng)
}

pub fn initialize_parameters_f64_constant(
    layers: &[usize],
    weights: &mut [f64],
    biases: &mut [f64],
    value: f64,
) -> Result<(), InitError> {
    let mut rng = SplitMix64::new(0);
    initialize_parameters_f64_impl(layers, weights, biases, InitKindF64::Constant(value), &mut rng)
}

fn random_uniform_symmetric_f32(rng: &mut SplitMix64, limit: f32) -> f32 {
    let u = rng.next_f32_01();
    (u * 2.0 - 1.0) * limit
}

fn random_uniform_symmetric_f64(rng: &mut SplitMix64, limit: f64) -> f64 {
    let u = rng.next_f64_01();
    (u * 2.0 - 1.0) * limit
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_f32_01(&mut self) -> f32 {
        let raw = (self.next_u64() >> 40) as u32;
        raw as f32 / 16777216.0
    }

    fn next_f64_01(&mut self) -> f64 {
        let raw = self.next_u64() >> 11;
        raw as f64 / 9007199254740992.0f64
    }
}
