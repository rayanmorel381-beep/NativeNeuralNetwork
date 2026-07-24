use super::kind::OptimizerError;
use crate::base::math::Float;
use crate::engine::runtime::{thread_parallel_for, SyncMutPtr};

const ADAMW_PARALLEL_THRESHOLD: usize = 8192;
const ADAMW_CHUNK: usize = 4096;

pub struct AdamwConfig<T: Float> {
    pub learning_rate: T,
    pub step: u32,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
    pub weight_decay: T,
}

fn validate_adamw<T: Float>(cfg: &AdamwConfig<T>, n: usize, ng: usize, nm: usize, nv: usize) -> Result<(), OptimizerError> {
    if cfg.step == 0 { return Err(OptimizerError::StepOverflow); }
    if !cfg.beta1.is_finite() || !cfg.beta2.is_finite() || !cfg.eps.is_finite() || !cfg.weight_decay.is_finite() {
        return Err(OptimizerError::InvalidHyperParams);
    }
    if cfg.beta1 < T::ZERO || cfg.beta1 >= T::ONE || cfg.beta2 < T::ZERO || cfg.beta2 >= T::ONE || cfg.eps <= T::ZERO {
        return Err(OptimizerError::InvalidHyperParams);
    }
    if n != ng || nm != n || nv != n { return Err(OptimizerError::ShapeMismatch); }
    Ok(())
}

fn step_adamw_impl<T: Float>(
    params: &mut [T],
    grads: &[T],
    m: &mut [T],
    v: &mut [T],
    cfg: &AdamwConfig<T>,
) -> Result<(), OptimizerError> {
    validate_adamw(cfg, params.len(), grads.len(), m.len(), v.len())?;
    let t = T::from_i32(cfg.step as i32);
    let bc1 = T::ONE - cfg.beta1.powf(t);
    let bc2 = T::ONE - cfg.beta2.powf(t);
    if bc1 <= T::ZERO || bc2 <= T::ZERO { return Err(OptimizerError::StepOverflow); }
    let one_minus_b1 = T::ONE - cfg.beta1;
    let one_minus_b2 = T::ONE - cfg.beta2;
    let lr = cfg.learning_rate;
    let wd = cfg.weight_decay;
    let beta1 = cfg.beta1;
    let beta2 = cfg.beta2;
    let eps = cfg.eps;
    for i in 0..params.len() {
        let g = grads[i] + wd * params[i];
        m[i] = beta1 * m[i] + one_minus_b1 * g;
        v[i] = beta2 * v[i] + one_minus_b2 * g * g;
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
    Ok(())
}

fn step_adamw_parallel<T: Float>(
    params: &mut [T],
    grads: &[T],
    m: &mut [T],
    v: &mut [T],
    cfg: &AdamwConfig<T>,
) -> Result<(), OptimizerError> {
    validate_adamw(cfg, params.len(), grads.len(), m.len(), v.len())?;
    let t = T::from_i32(cfg.step as i32);
    let bc1 = T::ONE - cfg.beta1.powf(t);
    let bc2 = T::ONE - cfg.beta2.powf(t);
    if bc1 <= T::ZERO || bc2 <= T::ZERO { return Err(OptimizerError::StepOverflow); }
    let one_minus_b1 = T::ONE - cfg.beta1;
    let one_minus_b2 = T::ONE - cfg.beta2;
    let lr = cfg.learning_rate;
    let wd = cfg.weight_decay;
    let beta1 = cfg.beta1;
    let beta2 = cfg.beta2;
    let eps = cfg.eps;
    let n = params.len();
    let nchunks = (n + ADAMW_CHUNK - 1) / ADAMW_CHUNK;
    let p_ptr = SyncMutPtr(params.as_mut_ptr());
    let g_ptr = SyncMutPtr(grads.as_ptr() as *mut T);
    let m_ptr = SyncMutPtr(m.as_mut_ptr());
    let v_ptr = SyncMutPtr(v.as_mut_ptr());
    thread_parallel_for(nchunks, &move |c| {
        let _ = (&p_ptr, &g_ptr, &m_ptr, &v_ptr);
        let start = c * ADAMW_CHUNK;
        let end = (start + ADAMW_CHUNK).min(n);
        for i in start..end {
            unsafe {
                let p = &mut *p_ptr.0.add(i);
                let g = *g_ptr.0.add(i) + wd * *p;
                let mi = &mut *m_ptr.0.add(i);
                let vi = &mut *v_ptr.0.add(i);
                *mi = beta1 * *mi + one_minus_b1 * g;
                *vi = beta2 * *vi + one_minus_b2 * g * g;
                *p -= lr * (*mi / bc1) / ((*vi / bc2).sqrt() + eps);
            }
        }
    });
    Ok(())
}

pub fn step_adamw<T: Float>(
    params: &mut [T],
    grads: &[T],
    m: &mut [T],
    v: &mut [T],
    cfg: AdamwConfig<T>,
) -> Result<(), OptimizerError> {
    if crate::engine::try_invoke_gpu_adamw::<T>(
        params, grads, m, v,
        cfg.learning_rate, cfg.step, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay,
    ) {
        return Ok(());
    }
    if params.len() >= ADAMW_PARALLEL_THRESHOLD {
        return step_adamw_parallel(params, grads, m, v, &cfg);
    }
    step_adamw_impl(params, grads, m, v, &cfg)
}
