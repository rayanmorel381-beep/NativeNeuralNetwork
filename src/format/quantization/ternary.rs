use super::qtypes::QuantError;
use crate::base::math::Float;
use crate::engine::runtime::{parallel_for, SyncMutPtr};

const TRITS_PER_BYTE: usize = 5;

pub fn ternary_packed_len(n: usize) -> usize {
    n.div_ceil(TRITS_PER_BYTE)
}

fn encode_trit<T: Float>(v: T, threshold: T) -> u8 {
    if v > threshold {
        2
    } else if v < -threshold {
        0
    } else {
        1
    }
}

fn quantize_ternary_impl<T: Float>(input: &[T], packed: &mut [u8]) -> Result<T, QuantError> {
    if input.is_empty() {
        return Err(QuantError::Empty);
    }
    let needed = ternary_packed_len(input.len());
    if packed.len() < needed {
        return Err(QuantError::ShapeMismatch);
    }

    let mut abs_sum = T::ZERO;
    for &v in input {
        abs_sum += v.abs();
    }
    let mean_abs = abs_sum / T::from_usize(input.len());
    let scale = if mean_abs <= T::ZERO { T::ONE } else { mean_abs };
    let threshold = T::from_f32(0.7) * mean_abs;

    let n = input.len();
    let full_groups = n / TRITS_PER_BYTE;
    let remainder = n % TRITS_PER_BYTE;

    for (g, slot) in packed.iter_mut().enumerate().take(full_groups) {
        let base = g * TRITS_PER_BYTE;
        let mut code: u8 = 0;
        let mut mul: u8 = 1;
        for k in 0..TRITS_PER_BYTE {
            let trit = encode_trit(input[base + k], threshold);
            code = code.wrapping_add(trit.wrapping_mul(mul));
            mul = mul.wrapping_mul(3);
        }
        *slot = code;
    }

    if remainder > 0 {
        let base = full_groups * TRITS_PER_BYTE;
        let mut code: u8 = 0;
        let mut mul: u8 = 1;
        for k in 0..remainder {
            let trit = encode_trit(input[base + k], threshold);
            code = code.wrapping_add(trit.wrapping_mul(mul));
            mul = mul.wrapping_mul(3);
        }
        packed[full_groups] = code;
    }

    Ok(scale)
}

pub fn quantize_ternary<T: Float>(input: &[T], packed: &mut [u8]) -> Result<T, QuantError> {
    quantize_ternary_impl(input, packed)
}

pub fn matvec_ternary<T: Float>(
    inp: &[T],
    w_packed: &[u8],
    w_scales: &[T],
    in_d: usize,
    out_d: usize,
    out: &mut [T],
) -> Result<(), QuantError> {
    if in_d == 0 || out_d == 0 {
        return Err(QuantError::Empty);
    }
    let row_bytes = ternary_packed_len(in_d);
    let total = row_bytes.checked_mul(out_d).ok_or(QuantError::ShapeMismatch)?;
    if inp.len() < in_d || w_packed.len() < total || w_scales.len() < out_d || out.len() < out_d {
        return Err(QuantError::ShapeMismatch);
    }
    let out_ptr = SyncMutPtr(out.as_mut_ptr());
    parallel_for(out_d, in_d, &move |o| {
        let _ = &out_ptr;
        let row = &w_packed[o * row_bytes..o * row_bytes + row_bytes];
        let full_groups = in_d / TRITS_PER_BYTE;
        let remainder = in_d % TRITS_PER_BYTE;
        let mut acc = T::ZERO;
        let mut idx = 0usize;
        for &byte in row.iter().take(full_groups) {
            let mut code = byte;
            for _ in 0..TRITS_PER_BYTE {
                let trit = (code % 3) as i32 - 1;
                acc += inp[idx] * T::from_i32(trit);
                code /= 3;
                idx += 1;
            }
        }
        if remainder > 0 {
            let mut code = row[full_groups];
            for _ in 0..remainder {
                let trit = (code % 3) as i32 - 1;
                acc += inp[idx] * T::from_i32(trit);
                code /= 3;
                idx += 1;
            }
        }
        unsafe {
            *out_ptr.0.add(o) = acc * w_scales[o];
        }
    });
    Ok(())
}
