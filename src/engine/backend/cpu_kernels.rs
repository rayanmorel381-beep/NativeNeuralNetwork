use super::contract::{register_cpu_kernel_f32, register_cpu_kernel_f64};
use super::types::{KernelInvokeF32, KernelInvokeF64};
use crate::engine::runtime::{parallel_for, SyncMutPtr, dot_f32};

fn cpu_kernel_f32(args: KernelInvokeF32<'_>) -> bool {
    let KernelInvokeF32 {
        src,
        dst,
        batch_size,
        stride,
        in_size,
        out_size,
        weights,
        biases,
        activation,
    } = args;
    if batch_size == 0 || in_size == 0 || out_size == 0 {
        return false;
    }
    if weights.len() < out_size * in_size {
        return false;
    }
    if batch_size > 1 && stride < out_size {
        return false;
    }
    let last_base = (batch_size - 1) * stride;
    if src.len() < last_base + in_size || dst.len() < last_base + out_size {
        return false;
    }
    let has_bias = biases.len() >= out_size;
    let dst_ptr = SyncMutPtr(dst.as_mut_ptr());
    parallel_for(out_size, in_size.saturating_mul(batch_size), &move |o| {
        let _ = &dst_ptr;
        let row = &weights[o * in_size..(o + 1) * in_size];
        let bias = if has_bias { biases[o] } else { 0.0 };
        let mut b = 0usize;
        while b < batch_size {
            let base = b * stride;
            let input = &src[base..base + in_size];
            let acc = dot_f32(input, row, in_size) + bias;
            unsafe {
                *dst_ptr.0.add(base + o) = activation.apply(acc);
            }
            b += 1;
        }
    });
    true
}

fn dot_f64_8acc(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut acc0 = 0.0f64;
    let mut acc1 = 0.0f64;
    let mut acc2 = 0.0f64;
    let mut acc3 = 0.0f64;
    let mut acc4 = 0.0f64;
    let mut acc5 = 0.0f64;
    let mut acc6 = 0.0f64;
    let mut acc7 = 0.0f64;
    let mut i = 0usize;
    while i + 8 <= n {
        acc0 += a[i] * b[i];
        acc1 += a[i + 1] * b[i + 1];
        acc2 += a[i + 2] * b[i + 2];
        acc3 += a[i + 3] * b[i + 3];
        acc4 += a[i + 4] * b[i + 4];
        acc5 += a[i + 5] * b[i + 5];
        acc6 += a[i + 6] * b[i + 6];
        acc7 += a[i + 7] * b[i + 7];
        i += 8;
    }
    let mut acc = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    while i < n {
        acc += a[i] * b[i];
        i += 1;
    }
    acc
}

fn cpu_kernel_f64(args: KernelInvokeF64<'_>) -> bool {
    let KernelInvokeF64 {
        src,
        dst,
        batch_size,
        stride,
        in_size,
        out_size,
        weights,
        biases,
        activation,
    } = args;
    if batch_size == 0 || in_size == 0 || out_size == 0 {
        return false;
    }
    if weights.len() < out_size * in_size {
        return false;
    }
    if batch_size > 1 && stride < out_size {
        return false;
    }
    let last_base = (batch_size - 1) * stride;
    if src.len() < last_base + in_size || dst.len() < last_base + out_size {
        return false;
    }
    let has_bias = biases.len() >= out_size;
    let dst_ptr = SyncMutPtr(dst.as_mut_ptr());
    parallel_for(out_size, in_size.saturating_mul(batch_size), &move |o| {
        let _ = &dst_ptr;
        let row = &weights[o * in_size..(o + 1) * in_size];
        let bias = if has_bias { biases[o] } else { 0.0 };
        let mut b = 0usize;
        while b < batch_size {
            let base = b * stride;
            let input = &src[base..base + in_size];
            let acc = dot_f64_8acc(input, row, in_size) + bias;
            unsafe {
                *dst_ptr.0.add(base + o) = activation.apply_f64(acc);
            }
            b += 1;
        }
    });
    true
}

pub(crate) fn install() {
    register_cpu_kernel_f32(cpu_kernel_f32);
    register_cpu_kernel_f64(cpu_kernel_f64);
}
