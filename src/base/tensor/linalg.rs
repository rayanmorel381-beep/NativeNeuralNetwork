use crate::base::math::Float;
use crate::engine::runtime::axpy;

pub fn tensor_trace<T: Float>(a: &[T], n: usize) -> T {
    if a.len() < n * n {
        return T::ZERO;
    }
    let mut sum = T::ZERO;
    for i in 0..n {
        sum += a[i * n + i];
    }
    sum
}

pub fn tensor_matmul_2d<T: Float>(
    a: &[T],
    b: &[T],
    c: &mut [T],
    m: usize,
    k: usize,
    n: usize,
) -> bool {
    if a.len() < m * k || b.len() < k * n || c.len() < m * n {
        return false;
    }
    for v in c[..m * n].iter_mut() {
        *v = T::ZERO;
    }
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            if a_ik.abs() > T::ZERO {
                axpy(&mut c[i * n..(i + 1) * n], a_ik, &b[kk * n..(kk + 1) * n], n);
            }
        }
    }
    true
}

pub fn tensor_transpose_2d<T: Float>(a: &[T], out: &mut [T], m: usize, n: usize) -> bool {
    if a.len() < m * n || out.len() < m * n {
        return false;
    }
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    true
}

pub fn tensor_solve_lower<T: Float>(l: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
    if l.len() < n * n || b.len() < n || x.len() < n {
        return false;
    }
    for i in 0..n {
        let mut val = b[i];
        for j in 0..i {
            val -= l[i * n + j] * x[j];
        }
        let lii = l[i * n + i];
        if lii.abs() < T::from_f32(1e-30) {
            return false;
        }
        x[i] = val / lii;
    }
    true
}

pub fn tensor_solve_upper<T: Float>(u: &[T], b: &[T], x: &mut [T], n: usize) -> bool {
    if u.len() < n * n || b.len() < n || x.len() < n {
        return false;
    }
    for i in (0..n).rev() {
        let mut val = b[i];
        for j in i + 1..n {
            val -= u[i * n + j] * x[j];
        }
        let uii = u[i * n + i];
        if uii.abs() < T::from_f32(1e-30) {
            return false;
        }
        x[i] = val / uii;
    }
    true
}
