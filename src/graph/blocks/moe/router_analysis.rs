use crate::base::math::Float;
use crate::base::tensor::{
    tensor_matmul_2d, tensor_trace, tensor_transpose_2d,
    tensor_determinant, tensor_cholesky, tensor_inverse,
    tensor_eigenvalues_qr, tensor_eigenvectors_qr,
};

const MAX_ITER: usize = 64;

pub fn moe_router_condition<T: Float>(
    router: &[T],
    ne: usize,
    h: usize,
    scratch: &mut [T],
) -> T {
    if ne == 0 || h == 0 || router.len() < ne * h || scratch.len() < 4 * ne * ne + ne * h + ne {
        return T::ZERO;
    }
    let (gram, rest) = scratch.split_at_mut(ne * ne);
    let (rt, rest2) = rest.split_at_mut(ne * h);
    let (eigenvals, qr_scratch) = rest2.split_at_mut(ne);
    tensor_transpose_2d(router, rt, h, ne);
    tensor_matmul_2d(router, rt, gram, ne, h, ne);
    let tr = tensor_trace(gram, ne);
    if !tensor_eigenvalues_qr(gram, ne, MAX_ITER, T::from_f32(1e-6), eigenvals, qr_scratch) {
        return tr;
    }
    let mut max_e = T::ZERO;
    let mut min_e = tr;
    for i in 0..ne {
        if eigenvals[i] > max_e { max_e = eigenvals[i]; }
        if eigenvals[i] < min_e { min_e = eigenvals[i]; }
    }
    if min_e <= T::ZERO { return tr; }
    max_e / min_e
}

pub fn moe_router_is_pd<T: Float>(
    router: &[T],
    ne: usize,
    h: usize,
    scratch: &mut [T],
) -> bool {
    if ne == 0 || h == 0 || router.len() < ne * h || scratch.len() < 2 * ne * ne + ne * h {
        return false;
    }
    let (gram, rest) = scratch.split_at_mut(ne * ne);
    let (rt, chol) = rest.split_at_mut(ne * h);
    if chol.len() < ne * ne { return false; }
    tensor_transpose_2d(router, rt, h, ne);
    tensor_matmul_2d(router, rt, gram, ne, h, ne);
    tensor_cholesky(gram, ne, chol)
}

pub fn moe_router_inverse_gram<T: Float>(
    router: &[T],
    ne: usize,
    h: usize,
    out: &mut [T],
    scratch: &mut [T],
) -> bool {
    if ne == 0 || h == 0 || router.len() < ne * h || out.len() < ne * ne
        || scratch.len() < ne * h + ne * ne + 2 * ne * ne + 2 * ne {
        return false;
    }
    let (rt, rest) = scratch.split_at_mut(ne * h);
    let (gram, inv_scratch) = rest.split_at_mut(ne * ne);
    tensor_transpose_2d(router, rt, h, ne);
    tensor_matmul_2d(router, rt, gram, ne, h, ne);
    tensor_inverse(gram, ne, out, inv_scratch)
}

pub fn moe_router_eigenvectors<T: Float>(
    router: &[T],
    ne: usize,
    h: usize,
    eigenvals: &mut [T],
    eigenvecs: &mut [T],
    scratch: &mut [T],
) -> bool {
    if ne == 0 || h == 0 || router.len() < ne * h || eigenvals.len() < ne
        || eigenvecs.len() < ne * ne || scratch.len() < ne * h + ne * ne + 5 * ne * ne + ne {
        return false;
    }
    let (rt, rest) = scratch.split_at_mut(ne * h);
    let (gram, qr_scratch) = rest.split_at_mut(ne * ne);
    tensor_transpose_2d(router, rt, h, ne);
    tensor_matmul_2d(router, rt, gram, ne, h, ne);
    tensor_eigenvectors_qr(gram, ne, MAX_ITER, T::from_f32(1e-6), eigenvals, eigenvecs, qr_scratch)
}

pub fn moe_router_determinant<T: Float>(
    router: &[T],
    ne: usize,
    h: usize,
    scratch: &mut [T],
) -> T {
    if ne == 0 || h == 0 || router.len() < ne * h || scratch.len() < ne * h + 3 * ne * ne {
        return T::ZERO;
    }
    let (rt, rest) = scratch.split_at_mut(ne * h);
    let (gram, det_scratch) = rest.split_at_mut(ne * ne);
    tensor_transpose_2d(router, rt, h, ne);
    tensor_matmul_2d(router, rt, gram, ne, h, ne);
    tensor_determinant(gram, ne, det_scratch).unwrap_or(T::ZERO)
}
