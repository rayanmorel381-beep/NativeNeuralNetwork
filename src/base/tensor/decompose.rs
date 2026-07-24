use crate::base::math::Float;
use super::linalg::{tensor_matmul_2d, tensor_solve_upper, tensor_solve_lower, tensor_trace, tensor_transpose_2d};

pub fn tensor_lu<T: Float>(a: &[T], n: usize, l: &mut [T], u: &mut [T]) -> bool {
    if a.len() < n * n || l.len() < n * n || u.len() < n * n {
        return false;
    }
    for i in 0..n {
        for j in 0..n {
            l[i * n + j] = if i == j { T::ONE } else { T::ZERO };
            u[i * n + j] = a[i * n + j];
        }
    }
    for j in 0..n {
        for i in j + 1..n {
            let pivot = u[j * n + j];
            if pivot.abs() < T::from_f32(1e-30) {
                continue;
            }
            let factor = u[i * n + j] / pivot;
            l[i * n + j] = factor;
            for k in j..n {
                let val = u[i * n + k] - factor * u[j * n + k];
                u[i * n + k] = val;
            }
        }
    }
    true
}

pub fn tensor_cholesky<T: Float>(a: &[T], n: usize, l: &mut [T]) -> bool {
    if a.len() < n * n || l.len() < n * n {
        return false;
    }
    for v in l[..n * n].iter_mut() {
        *v = T::ZERO;
    }
    for i in 0..n {
        for j in 0..=i {
            let mut sum = T::ZERO;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let val = a[i * n + i] - sum;
                if val <= T::ZERO {
                    return false;
                }
                l[i * n + j] = val.sqrt();
            } else {
                let ljj = l[j * n + j];
                if ljj.abs() < T::from_f32(1e-30) {
                    return false;
                }
                l[i * n + j] = (a[i * n + j] - sum) / ljj;
            }
        }
    }
    true
}

pub fn tensor_qr<T: Float>(
    a: &[T],
    m: usize,
    n: usize,
    q: &mut [T],
    r: &mut [T],
    scratch: &mut [T],
) -> bool {
    if a.len() < m * n || q.len() < m * n || r.len() < n * n || scratch.len() < m {
        return false;
    }
    for v in q[..m * n].iter_mut() {
        *v = T::ZERO;
    }
    for j in 0..n {
        for i in 0..m {
            scratch[i] = a[i * n + j];
        }
        for k in 0..j {
            let mut dot = T::ZERO;
            for i in 0..m {
                dot += scratch[i] * q[i * n + k];
            }
            for i in 0..m {
                scratch[i] -= dot * q[i * n + k];
            }
        }
        let mut norm_sq = T::ZERO;
        for i in 0..m {
            norm_sq += scratch[i] * scratch[i];
        }
        let norm = norm_sq.sqrt();
        if norm > T::from_f32(1e-15) {
            for i in 0..m {
                q[i * n + j] = scratch[i] / norm;
            }
        }
    }
    for k in 0..n {
        for j in 0..n {
            let mut sum = T::ZERO;
            for i in 0..m {
                sum += q[i * n + k] * a[i * n + j];
            }
            r[k * n + j] = sum;
        }
    }
    true
}

pub fn tensor_determinant<T: Float>(
    a: &[T],
    n: usize,
    scratch: &mut [T],
) -> Option<T> {
    if a.len() < n * n || scratch.len() < 2 * n * n {
        return None;
    }
    let (l, u) = scratch.split_at_mut(n * n);
    if !tensor_lu(a, n, l, u) {
        return None;
    }
    let mut det = T::ONE;
    for i in 0..n {
        det *= u[i * n + i];
    }
    Some(det)
}

pub fn tensor_inverse<T: Float>(
    a: &[T],
    n: usize,
    out: &mut [T],
    scratch: &mut [T],
) -> bool {
    if a.len() < n * n || out.len() < n * n || scratch.len() < 2 * n * n + 2 * n {
        return false;
    }
    let (l, rest) = scratch.split_at_mut(n * n);
    let (u, rest) = rest.split_at_mut(n * n);
    let (y_buf, x_buf) = rest.split_at_mut(n);

    if !tensor_lu(a, n, l, u) {
        return false;
    }
    for col in 0..n {
        for i in 0..n {
            y_buf[i] = T::ZERO;
        }
        y_buf[col] = T::ONE;
        if !tensor_solve_lower(l, y_buf, x_buf, n) {
            return false;
        }
        if !tensor_solve_upper(u, x_buf, y_buf, n) {
            return false;
        }
        for i in 0..n {
            out[i * n + col] = y_buf[i];
        }
    }
    true
}

pub fn tensor_eigenvalues_qr<T: Float>(
    a: &[T],
    n: usize,
    max_iter: usize,
    tol: T,
    out: &mut [T],
    scratch: &mut [T],
) -> bool {
    if a.len() < n * n || out.len() < n || scratch.len() < 4 * n * n + n {
        return false;
    }
    let (ak, rest) = scratch.split_at_mut(n * n);
    let (q_buf, rest) = rest.split_at_mut(n * n);
    let (r_buf, rest) = rest.split_at_mut(n * n);
    let (tmp, gs) = rest.split_at_mut(n * n);

    ak[..n * n].copy_from_slice(&a[..n * n]);

    for _ in 0..max_iter {
        if !tensor_qr(ak, n, n, q_buf, r_buf, gs) {
            return false;
        }
        for v in tmp[..n * n].iter_mut() {
            *v = T::ZERO;
        }
        tensor_matmul_2d(r_buf, q_buf, tmp, n, n, n);
        ak[..n * n].copy_from_slice(&tmp[..n * n]);

        let mut off_diag = T::ZERO;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let v = ak[i * n + j];
                    off_diag += v * v;
                }
            }
        }
        if off_diag.sqrt() < tol {
            break;
        }
    }
    for i in 0..n {
        out[i] = ak[i * n + i];
    }
    let tr = tensor_trace(ak, n);
    let sum: T = out.iter().take(n).copied().fold(T::ZERO, |s, e| s + e);
    if n > 0 {
        let correction = (tr - sum) / T::from_usize(n);
        for v in out.iter_mut().take(n) {
            *v += correction;
        }
    }
    true
}

pub fn tensor_eigenvectors_qr<T: Float>(
    a: &[T],
    n: usize,
    max_iter: usize,
    tol: T,
    eigenvals: &mut [T],
    eigenvecs: &mut [T],
    scratch: &mut [T],
) -> bool {
    if a.len() < n * n
        || eigenvals.len() < n
        || eigenvecs.len() < n * n
        || scratch.len() < 5 * n * n + n
    {
        return false;
    }
    let (ak, rest) = scratch.split_at_mut(n * n);
    let (v_acc, rest) = rest.split_at_mut(n * n);
    let (q_buf, rest) = rest.split_at_mut(n * n);
    let (r_buf, rest) = rest.split_at_mut(n * n);
    let (tmp, gs) = rest.split_at_mut(n * n);

    ak[..n * n].copy_from_slice(&a[..n * n]);
    for i in 0..n {
        for j in 0..n {
            v_acc[i * n + j] = if i == j { T::ONE } else { T::ZERO };
        }
    }

    for _ in 0..max_iter {
        if !tensor_qr(ak, n, n, q_buf, r_buf, gs) {
            return false;
        }
        for v in tmp[..n * n].iter_mut() {
            *v = T::ZERO;
        }
        tensor_matmul_2d(r_buf, q_buf, tmp, n, n, n);
        ak[..n * n].copy_from_slice(&tmp[..n * n]);

        for v in tmp[..n * n].iter_mut() {
            *v = T::ZERO;
        }
        tensor_matmul_2d(v_acc, q_buf, tmp, n, n, n);
        v_acc[..n * n].copy_from_slice(&tmp[..n * n]);

        let mut off_diag = T::ZERO;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let v = ak[i * n + j];
                    off_diag += v * v;
                }
            }
        }
        if off_diag.sqrt() < tol {
            break;
        }
    }
    for i in 0..n {
        eigenvals[i] = ak[i * n + i];
    }
    tensor_transpose_2d(&v_acc[..n * n], eigenvecs, n, n);
    true
}
