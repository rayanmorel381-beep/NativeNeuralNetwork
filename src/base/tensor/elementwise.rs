use super::TensorView;
use crate::base::math::Float;

pub fn tensor_fill(tensor: &mut TensorView<'_>, value: f32) {
    for v in tensor.data.iter_mut() {
        *v = value;
    }
}

pub fn tensor_scale_in_place(tensor: &mut TensorView<'_>, scale: f32) {
    for v in tensor.data.iter_mut() {
        *v *= scale;
    }
}

pub fn tensor_add_in_place(dst: &mut TensorView<'_>, src: &TensorView<'_>) -> bool {
    if dst.shape != src.shape || dst.data.len() != src.data.len() {
        return false;
    }
    for i in 0..dst.data.len() {
        dst.data[i] += src.data[i];
    }
    true
}

pub fn tensor_relu(tensor: &mut TensorView<'_>) {
    for v in tensor.data.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

pub fn tensor_add_bias(tensor: &mut TensorView<'_>, bias: &[f32]) -> bool {
    let [n, c, d, h, w] = tensor.shape;
    if w == 0 || bias.len() < w {
        return false;
    }
    let rows = n * c * d * h;
    for r in 0..rows {
        let row = &mut tensor.data[r * w..(r + 1) * w];
        for (v, &b) in row.iter_mut().zip(bias.iter()) {
            *v += b;
        }
    }
    true
}

pub fn tensor_softmax_rows(tensor: &mut TensorView<'_>) {
    let [n, c, d, h, w] = tensor.shape;
    if w == 0 {
        return;
    }
    let rows = n * c * d * h;
    for r in 0..rows {
        let row = &mut tensor.data[r * w..(r + 1) * w];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = Float::exp(*v - max);
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

pub fn tensor_matmul_add(
    a: &TensorView<'_>,
    b: &TensorView<'_>,
    out: &mut TensorView<'_>,
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    if a.data.len() < m * k || b.data.len() < k * n || out.data.len() < m * n {
        return false;
    }
    for i in 0..m {
        let ar = &a.data[i * k..i * k + k];
        for j in 0..n {
            let br_col: f32 = (0..k).map(|p| ar[p] * b.data[p * n + j]).sum();
            out.data[i * n + j] += br_col;
        }
    }
    true
}
