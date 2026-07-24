use crate::base::activations::ActivationKind;
use crate::base::math::Float;

fn dot_unrolled<T: Float>(weights: &[T], input: &[T]) -> T {
    let len = weights.len();
    let mut i = 0usize;
    let mut acc0 = T::ZERO;
    let mut acc1 = T::ZERO;
    let mut acc2 = T::ZERO;
    let mut acc3 = T::ZERO;

    while i + 4 <= len {
        acc0 += weights[i] * input[i];
        acc1 += weights[i + 1] * input[i + 1];
        acc2 += weights[i + 2] * input[i + 2];
        acc3 += weights[i + 3] * input[i + 3];
        i += 4;
    }

    let mut acc = (acc0 + acc1) + (acc2 + acc3);
    while i < len {
        acc += weights[i] * input[i];
        i += 1;
    }
    acc
}

pub(crate) fn dense_layer_forward<T: Float>(
    input: &[T],
    weights: &[T],
    biases: &[T],
    in_size: usize,
    out_size: usize,
    activation: ActivationKind,
    out: &mut [T],
) {
    if crate::engine::try_invoke_gpu_kernel::<T>(
        input,
        out,
        1,
        in_size,
        in_size,
        out_size,
        weights,
        biases,
        activation,
    ) {
        return;
    }
    for o in 0..out_size {
        let row = &weights[o * in_size..o * in_size + in_size];
        let acc = biases[o] + dot_unrolled(row, &input[..in_size]);
        out[o] = activation.apply_t(acc);
    }
}
