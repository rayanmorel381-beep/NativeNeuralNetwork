use super::types::*;
use super::contract::{
    decode_cpu_kernel_f32, decode_cpu_kernel_f64, dispatch_contract, get_compute_backend,
    CPU_KERNEL_F32, CPU_KERNEL_F64,
};
use core::sync::atomic::Ordering;

pub(crate) fn try_invoke_gpu_kernel_f32(args: KernelInvokeF32<'_>) -> bool {
    let KernelInvokeF32 {
        src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
    } = args;
    if batch_size > 1 && get_compute_backend() == ComputeBackend::Gpu {
        let cmd = KernelCmdF32 {
            src_ptr: src.as_ptr(),
            src_len: src.len(),
            dst_ptr: dst.as_mut_ptr(),
            dst_len: dst.len(),
            batch_size,
            stride,
            in_size,
            out_size,
            weights_ptr: weights.as_ptr(),
            weights_len: weights.len(),
            biases_ptr: biases.as_ptr(),
            biases_len: biases.len(),
            activation: activation.to_u8(),
        };
        if dispatch_contract(
            GpuOpCode::F32,
            (&cmd as *const KernelCmdF32).cast::<u8>(),
            core::mem::size_of::<KernelCmdF32>(),
        ) {
            return true;
        }
    }
    if let Some(kernel) = decode_cpu_kernel_f32(CPU_KERNEL_F32.load(Ordering::SeqCst)) {
        return kernel(KernelInvokeF32 {
            src,
            dst,
            batch_size,
            stride,
            in_size,
            out_size,
            weights,
            biases,
            activation,
        });
    }
    false
}

pub(crate) fn try_invoke_gpu_kernel_f64(args: KernelInvokeF64<'_>) -> bool {
    let KernelInvokeF64 {
        src, dst, batch_size, stride, in_size, out_size, weights, biases, activation,
    } = args;
    if batch_size > 1 && get_compute_backend() == ComputeBackend::Gpu {
        let cmd = KernelCmdF64 {
            src_ptr: src.as_ptr(),
            src_len: src.len(),
            dst_ptr: dst.as_mut_ptr(),
            dst_len: dst.len(),
            batch_size,
            stride,
            in_size,
            out_size,
            weights_ptr: weights.as_ptr(),
            weights_len: weights.len(),
            biases_ptr: biases.as_ptr(),
            biases_len: biases.len(),
            activation: activation.to_u8(),
        };
        if dispatch_contract(
            GpuOpCode::F64,
            (&cmd as *const KernelCmdF64).cast::<u8>(),
            core::mem::size_of::<KernelCmdF64>(),
        ) {
            return true;
        }
    }
    if let Some(kernel) = decode_cpu_kernel_f64(CPU_KERNEL_F64.load(Ordering::SeqCst)) {
        return kernel(KernelInvokeF64 {
            src,
            dst,
            batch_size,
            stride,
            in_size,
            out_size,
            weights,
            biases,
            activation,
        });
    }
    false
}

pub(crate) fn try_invoke_gpu_softmax_f32(logits: &[f32], out: &mut [f32]) -> bool {
    let cmd = SoftmaxCmdF32 {
        logits_ptr: logits.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        len: logits.len(),
    };
    dispatch_contract(
        GpuOpCode::SoftmaxF32,
        (&cmd as *const SoftmaxCmdF32).cast::<u8>(),
        core::mem::size_of::<SoftmaxCmdF32>(),
    )
}

pub(crate) fn try_invoke_gpu_softmax_f64(logits: &[f64], out: &mut [f64]) -> bool {
    let cmd = SoftmaxCmdF64 {
        logits_ptr: logits.as_ptr(),
        out_ptr: out.as_mut_ptr(),
        len: logits.len(),
    };
    dispatch_contract(
        GpuOpCode::SoftmaxF64,
        (&cmd as *const SoftmaxCmdF64).cast::<u8>(),
        core::mem::size_of::<SoftmaxCmdF64>(),
    )
}

pub(crate) fn try_invoke_gpu_rms_norm_f32(x: &mut [f32], gamma: &[f32], eps: f32) -> bool {
    let cmd = RmsNormCmdF32 {
        x_ptr: x.as_mut_ptr(),
        gamma_ptr: gamma.as_ptr(),
        len: x.len(),
        eps,
    };
    dispatch_contract(
        GpuOpCode::RmsNormF32,
        (&cmd as *const RmsNormCmdF32).cast::<u8>(),
        core::mem::size_of::<RmsNormCmdF32>(),
    )
}

pub(crate) fn try_invoke_gpu_rms_norm_f64(x: &mut [f64], gamma: &[f64], eps: f64) -> bool {
    let cmd = RmsNormCmdF64 {
        x_ptr: x.as_mut_ptr(),
        gamma_ptr: gamma.as_ptr(),
        len: x.len(),
        eps,
    };
    dispatch_contract(
        GpuOpCode::RmsNormF64,
        (&cmd as *const RmsNormCmdF64).cast::<u8>(),
        core::mem::size_of::<RmsNormCmdF64>(),
    )
}

pub fn try_invoke_gpu_attention_f32(args: AttentionInvokeF32<'_>) -> bool {
    let AttentionInvokeF32 {
        q, k, v, out, scratch_scores, q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
    } = args;
    let cmd = AttentionCmdF32 {
        q_ptr: q.as_ptr(),
        q_len_total: q.len(),
        k_ptr: k.as_ptr(),
        k_len_total: k.len(),
        v_ptr: v.as_ptr(),
        v_len_total: v.len(),
        out_ptr: out.as_mut_ptr(),
        out_len_total: out.len(),
        scratch_scores_ptr: scratch_scores.as_mut_ptr(),
        scratch_scores_len: scratch_scores.len(),
        q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
    };
    dispatch_contract(
        GpuOpCode::AttentionF32,
        (&cmd as *const AttentionCmdF32).cast::<u8>(),
        core::mem::size_of::<AttentionCmdF32>(),
    )
}

pub(crate) fn try_invoke_gpu_attention_f64(args: AttentionInvokeF64<'_>) -> bool {
    let AttentionInvokeF64 {
        q, k, v, out, scratch_scores, q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
    } = args;
    let cmd = AttentionCmdF64 {
        q_ptr: q.as_ptr(),
        q_len_total: q.len(),
        k_ptr: k.as_ptr(),
        k_len_total: k.len(),
        v_ptr: v.as_ptr(),
        v_len_total: v.len(),
        out_ptr: out.as_mut_ptr(),
        out_len_total: out.len(),
        scratch_scores_ptr: scratch_scores.as_mut_ptr(),
        scratch_scores_len: scratch_scores.len(),
        q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
    };
    dispatch_contract(
        GpuOpCode::AttentionF64,
        (&cmd as *const AttentionCmdF64).cast::<u8>(),
        core::mem::size_of::<AttentionCmdF64>(),
    )
}

pub(crate) fn try_invoke_gpu_quantize_i8_f32(
    input: &[f32], output: &mut [i8], scale_out: &mut f32,
) -> bool {
    let cmd = QuantizeI8CmdF32 {
        input_ptr: input.as_ptr(),
        input_len: input.len(),
        output_ptr: output.as_mut_ptr(),
        output_len: output.len(),
        scale_out_ptr: scale_out as *mut f32,
    };
    dispatch_contract(
        GpuOpCode::QuantizeI8F32,
        (&cmd as *const QuantizeI8CmdF32).cast::<u8>(),
        core::mem::size_of::<QuantizeI8CmdF32>(),
    )
}

pub(crate) fn try_invoke_gpu_quantize_i8<T: crate::base::math::Float>(
    input: &[T], output: &mut [i8], scale_out: &mut T,
) -> bool {
    match T::DTYPE_TAG {
        0 => {
            let i32s = unsafe { core::slice::from_raw_parts(input.as_ptr().cast::<f32>(), input.len()) };
            let mut s = 0.0f32;
            let ok = try_invoke_gpu_quantize_i8_f32(i32s, output, &mut s);
            if ok {
                *scale_out = T::from_f32(s);
            }
            ok
        }
        _ => false,
    }
}

pub fn try_invoke_gpu_sgd_f32(
    params: &mut [f32], grads: &[f32], velocity: &mut [f32],
    learning_rate: f32, momentum: f32, nesterov: bool,
) -> bool {
    let cmd = SgdCmdF32 {
        params_ptr: params.as_mut_ptr(),
        grads_ptr: grads.as_ptr(),
        velocity_ptr: velocity.as_mut_ptr(),
        len: params.len(),
        learning_rate, momentum,
        nesterov: if nesterov { 1 } else { 0 },
    };
    dispatch_contract(
        GpuOpCode::SgdF32,
        (&cmd as *const SgdCmdF32).cast::<u8>(),
        core::mem::size_of::<SgdCmdF32>(),
    )
}

pub fn try_invoke_gpu_sgd_f64(
    params: &mut [f64], grads: &[f64], velocity: &mut [f64],
    learning_rate: f64, momentum: f64, nesterov: bool,
) -> bool {
    let cmd = SgdCmdF64 {
        params_ptr: params.as_mut_ptr(),
        grads_ptr: grads.as_ptr(),
        velocity_ptr: velocity.as_mut_ptr(),
        len: params.len(),
        learning_rate, momentum,
        nesterov: if nesterov { 1 } else { 0 },
    };
    dispatch_contract(
        GpuOpCode::SgdF64,
        (&cmd as *const SgdCmdF64).cast::<u8>(),
        core::mem::size_of::<SgdCmdF64>(),
    )
}

pub(crate) fn try_invoke_gpu_adamw_f32(args: AdamwInvokeF32<'_>) -> bool {
    let AdamwInvokeF32 {
        params, grads, m, v, learning_rate, step, beta1, beta2, eps, weight_decay,
    } = args;
    let cmd = AdamwCmdF32 {
        params_ptr: params.as_mut_ptr(),
        grads_ptr: grads.as_ptr(),
        m_ptr: m.as_mut_ptr(),
        v_ptr: v.as_mut_ptr(),
        len: params.len(),
        learning_rate, step, beta1, beta2, eps, weight_decay,
    };
    dispatch_contract(
        GpuOpCode::AdamwF32,
        (&cmd as *const AdamwCmdF32).cast::<u8>(),
        core::mem::size_of::<AdamwCmdF32>(),
    )
}

pub(crate) fn try_invoke_gpu_adamw_f64(args: AdamwInvokeF64<'_>) -> bool {
    let AdamwInvokeF64 {
        params, grads, m, v, learning_rate, step, beta1, beta2, eps, weight_decay,
    } = args;
    let cmd = AdamwCmdF64 {
        params_ptr: params.as_mut_ptr(),
        grads_ptr: grads.as_ptr(),
        m_ptr: m.as_mut_ptr(),
        v_ptr: v.as_mut_ptr(),
        len: params.len(),
        learning_rate, step, beta1, beta2, eps, weight_decay,
    };
    dispatch_contract(
        GpuOpCode::AdamwF64,
        (&cmd as *const AdamwCmdF64).cast::<u8>(),
        core::mem::size_of::<AdamwCmdF64>(),
    )
}

pub(crate) fn try_invoke_gpu_rms_norm<T: crate::base::math::Float>(
    x: &mut [T], gamma: &[T], eps: T,
) -> bool {
    match T::DTYPE_TAG {
        0 => {
            let x32 = unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<f32>(), x.len()) };
            let g32 = unsafe { core::slice::from_raw_parts(gamma.as_ptr().cast::<f32>(), gamma.len()) };
            try_invoke_gpu_rms_norm_f32(x32, g32, eps.to_f64() as f32)
        }
        1 => {
            let x64 = unsafe { core::slice::from_raw_parts_mut(x.as_mut_ptr().cast::<f64>(), x.len()) };
            let g64 = unsafe { core::slice::from_raw_parts(gamma.as_ptr().cast::<f64>(), gamma.len()) };
            try_invoke_gpu_rms_norm_f64(x64, g64, eps.to_f64())
        }
        _ => false,
    }
}

pub(crate) fn try_invoke_gpu_softmax<T: crate::base::math::Float>(logits: &[T], out: &mut [T]) -> bool {
    match T::DTYPE_TAG {
        0 => {
            let l32 = unsafe { core::slice::from_raw_parts(logits.as_ptr().cast::<f32>(), logits.len()) };
            let o32 = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f32>(), out.len()) };
            try_invoke_gpu_softmax_f32(l32, o32)
        }
        1 => {
            let l64 = unsafe { core::slice::from_raw_parts(logits.as_ptr().cast::<f64>(), logits.len()) };
            let o64 = unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len()) };
            try_invoke_gpu_softmax_f64(l64, o64)
        }
        _ => false,
    }
}

pub(crate) fn try_invoke_gpu_attention<T: crate::base::math::Float>(
    args: AttentionInvoke<'_, T>,
) -> bool {
    let AttentionInvoke {
        q, k, v, out, scratch_scores, q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
    } = args;
    match T::DTYPE_TAG {
        0 => try_invoke_gpu_attention_f32(AttentionInvokeF32 {
            q: unsafe { core::slice::from_raw_parts(q.as_ptr().cast::<f32>(), q.len()) },
            k: unsafe { core::slice::from_raw_parts(k.as_ptr().cast::<f32>(), k.len()) },
            v: unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<f32>(), v.len()) },
            out: unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f32>(), out.len()) },
            scratch_scores: unsafe {
                core::slice::from_raw_parts_mut(scratch_scores.as_mut_ptr().cast::<f32>(), scratch_scores.len())
            },
            q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
        }),
        1 => try_invoke_gpu_attention_f64(AttentionInvokeF64 {
            q: unsafe { core::slice::from_raw_parts(q.as_ptr().cast::<f64>(), q.len()) },
            k: unsafe { core::slice::from_raw_parts(k.as_ptr().cast::<f64>(), k.len()) },
            v: unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<f64>(), v.len()) },
            out: unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<f64>(), out.len()) },
            scratch_scores: unsafe {
                core::slice::from_raw_parts_mut(scratch_scores.as_mut_ptr().cast::<f64>(), scratch_scores.len())
            },
            q_len, k_len, d_k, d_v, k_stride, v_stride, head_offset, mask,
        }),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_invoke_gpu_sgd<T: crate::base::math::Float>(
    params: &mut [T], grads: &[T], velocity: &mut [T],
    learning_rate: T, momentum: T, nesterov: bool,
) -> bool {
    match T::DTYPE_TAG {
        0 => {
            let p = unsafe { core::slice::from_raw_parts_mut(params.as_mut_ptr().cast::<f32>(), params.len()) };
            let g = unsafe { core::slice::from_raw_parts(grads.as_ptr().cast::<f32>(), grads.len()) };
            let ve = unsafe { core::slice::from_raw_parts_mut(velocity.as_mut_ptr().cast::<f32>(), velocity.len()) };
            try_invoke_gpu_sgd_f32(p, g, ve, learning_rate.to_f64() as f32, momentum.to_f64() as f32, nesterov)
        }
        1 => {
            let p = unsafe { core::slice::from_raw_parts_mut(params.as_mut_ptr().cast::<f64>(), params.len()) };
            let g = unsafe { core::slice::from_raw_parts(grads.as_ptr().cast::<f64>(), grads.len()) };
            let ve = unsafe { core::slice::from_raw_parts_mut(velocity.as_mut_ptr().cast::<f64>(), velocity.len()) };
            try_invoke_gpu_sgd_f64(p, g, ve, learning_rate.to_f64(), momentum.to_f64(), nesterov)
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_invoke_gpu_adamw<T: crate::base::math::Float>(
    params: &mut [T], grads: &[T], m: &mut [T], v: &mut [T],
    learning_rate: T, step: u32, beta1: T, beta2: T, eps: T, weight_decay: T,
) -> bool {
    let len = params.len();
    match T::DTYPE_TAG {
        0 => {
            let p = unsafe { core::slice::from_raw_parts_mut(params.as_mut_ptr().cast::<f32>(), len) };
            let g = unsafe { core::slice::from_raw_parts(grads.as_ptr().cast::<f32>(), grads.len()) };
            let mm = unsafe { core::slice::from_raw_parts_mut(m.as_mut_ptr().cast::<f32>(), m.len()) };
            let vv = unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f32>(), v.len()) };
            try_invoke_gpu_adamw_f32(AdamwInvokeF32 {
                params: p, grads: g, m: mm, v: vv,
                learning_rate: learning_rate.to_f64() as f32, step,
                beta1: beta1.to_f64() as f32, beta2: beta2.to_f64() as f32,
                eps: eps.to_f64() as f32, weight_decay: weight_decay.to_f64() as f32,
            })
        }
        1 => {
            let p = unsafe { core::slice::from_raw_parts_mut(params.as_mut_ptr().cast::<f64>(), len) };
            let g = unsafe { core::slice::from_raw_parts(grads.as_ptr().cast::<f64>(), grads.len()) };
            let mm = unsafe { core::slice::from_raw_parts_mut(m.as_mut_ptr().cast::<f64>(), m.len()) };
            let vv = unsafe { core::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<f64>(), v.len()) };
            try_invoke_gpu_adamw_f64(AdamwInvokeF64 {
                params: p, grads: g, m: mm, v: vv,
                learning_rate: learning_rate.to_f64(), step,
                beta1: beta1.to_f64(), beta2: beta2.to_f64(),
                eps: eps.to_f64(), weight_decay: weight_decay.to_f64(),
            })
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_invoke_gpu_kernel<T: crate::base::math::Float>(
    src: &[T], dst: &mut [T], batch_size: usize, stride: usize,
    in_size: usize, out_size: usize, weights: &[T], biases: &[T],
    activation: crate::base::activations::ActivationKind,
) -> bool {
    match T::DTYPE_TAG {
        0 => {
            let s = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<f32>(), src.len()) };
            let d = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f32>(), dst.len()) };
            let w = unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<f32>(), weights.len()) };
            let b = unsafe { core::slice::from_raw_parts(biases.as_ptr().cast::<f32>(), biases.len()) };
            try_invoke_gpu_kernel_f32(KernelInvokeF32 {
                src: s, dst: d, batch_size, stride, in_size, out_size,
                weights: w, biases: b, activation,
            })
        }
        1 => {
            let s = unsafe { core::slice::from_raw_parts(src.as_ptr().cast::<f64>(), src.len()) };
            let d = unsafe { core::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<f64>(), dst.len()) };
            let w = unsafe { core::slice::from_raw_parts(weights.as_ptr().cast::<f64>(), weights.len()) };
            let b = unsafe { core::slice::from_raw_parts(biases.as_ptr().cast::<f64>(), biases.len()) };
            try_invoke_gpu_kernel_f64(KernelInvokeF64 {
                src: s, dst: d, batch_size, stride, in_size, out_size,
                weights: w, biases: b, activation,
            })
        }
        _ => false,
    }
}
