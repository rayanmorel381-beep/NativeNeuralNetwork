use super::kind::OptimizerError;
use crate::base::math::Float;
use crate::engine::runtime::{thread_parallel_for, SyncMutPtr};

const SGD_PARALLEL_THRESHOLD: usize = 8192;
const SGD_CHUNK: usize = 4096;

fn step_sgd_impl<T: Float>(
    params: &mut [T],
    grads: &[T],
    velocity: &mut [T],
    learning_rate: T,
    momentum: T,
    nesterov: bool,
) -> Result<(), OptimizerError> {
    if !momentum.is_finite() || momentum < T::ZERO || momentum >= T::ONE {
        return Err(OptimizerError::InvalidHyperParams);
    }
    if params.len() != grads.len() || velocity.len() != params.len() {
        return Err(OptimizerError::ShapeMismatch);
    }
    for i in 0..params.len() {
        let v = momentum * velocity[i] + grads[i];
        velocity[i] = v;
        let update = if nesterov { momentum * v + grads[i] } else { v };
        params[i] -= learning_rate * update;
    }
    Ok(())
}

fn step_sgd_parallel<T: Float>(
    params: &mut [T],
    grads: &[T],
    velocity: &mut [T],
    learning_rate: T,
    momentum: T,
    nesterov: bool,
) -> Result<(), OptimizerError> {
    if !momentum.is_finite() || momentum < T::ZERO || momentum >= T::ONE {
        return Err(OptimizerError::InvalidHyperParams);
    }
    if params.len() != grads.len() || velocity.len() != params.len() {
        return Err(OptimizerError::ShapeMismatch);
    }
    let n = params.len();
    let nchunks = (n + SGD_CHUNK - 1) / SGD_CHUNK;
    let p_ptr = SyncMutPtr(params.as_mut_ptr());
    let g_ptr = SyncMutPtr(grads.as_ptr() as *mut T);
    let vel_ptr = SyncMutPtr(velocity.as_mut_ptr());
    thread_parallel_for(nchunks, &move |c| {
        let _ = (&p_ptr, &g_ptr, &vel_ptr);
        let start = c * SGD_CHUNK;
        let end = (start + SGD_CHUNK).min(n);
        for i in start..end {
            unsafe {
                let p = &mut *p_ptr.0.add(i);
                let g = *g_ptr.0.add(i);
                let vel = &mut *vel_ptr.0.add(i);
                let v = momentum * *vel + g;
                *vel = v;
                let upd = if nesterov { momentum * v + g } else { v };
                *p -= learning_rate * upd;
            }
        }
    });
    Ok(())
}

pub fn step_sgd<T: Float>(
    params: &mut [T],
    grads: &[T],
    velocity: &mut [T],
    learning_rate: T,
    momentum: T,
    nesterov: bool,
) -> Result<(), OptimizerError> {
    if crate::engine::try_invoke_gpu_sgd::<T>(params, grads, velocity, learning_rate, momentum, nesterov) {
        return Ok(());
    }
    if params.len() >= SGD_PARALLEL_THRESHOLD {
        return step_sgd_parallel(params, grads, velocity, learning_rate, momentum, nesterov);
    }
    step_sgd_impl(params, grads, velocity, learning_rate, momentum, nesterov)
}
