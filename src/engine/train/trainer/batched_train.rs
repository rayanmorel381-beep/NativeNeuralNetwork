use crate::engine::eval::losses::cross_entropy;
use crate::engine::train::optimizers::{step_adamw, AdamwConfig, OptimizerError};
use crate::engine::train::gradients::{clip_by_global_norm, l2_norm};
use crate::engine::rnn_flow::MAX_LAYERS;
use crate::engine::train::schedulers::compute_learning_rate;
use crate::engine::runtime::{parallel_for, SyncMutPtr};
use crate::engine::runtime::{axpy, dot};
use crate::base::math::Float;
use crate::graph::lm::{LmConfig, LmError, LmTrainConfig, LmTrainStep};
use crate::graph::lm::{LmBlockGrads, LmBlockWeights};

fn map_opt_err(e: OptimizerError) -> LmError {
    match e {
        OptimizerError::StepOverflow => LmError::InvalidConfig,
        OptimizerError::InvalidHyperParams => LmError::InvalidConfig,
        OptimizerError::ShapeMismatch => LmError::ShapeMismatch,
    }
}

const ELEMWISE_PARALLEL_THRESHOLD: usize = 1 << 16;

fn fill_zero<T: Float>(buf: &mut [T]) {
    let n = buf.len();
    if n < ELEMWISE_PARALLEL_THRESHOLD {
        for x in buf.iter_mut() {
            *x = T::ZERO;
        }
        return;
    }
    let ptr = SyncMutPtr(buf.as_mut_ptr());
    parallel_for(n, 1, &move |i| {
        let _ = &ptr;
        unsafe {
            *ptr.0.add(i) = T::ZERO;
        }
    });
}

fn scale_slice<T: Float>(buf: &mut [T], factor: T) {
    let n = buf.len();
    if n < ELEMWISE_PARALLEL_THRESHOLD {
        for x in buf.iter_mut() {
            *x *= factor;
        }
        return;
    }
    let ptr = SyncMutPtr(buf.as_mut_ptr());
    parallel_for(n, 1, &move |i| {
        let _ = &ptr;
        unsafe {
            *ptr.0.add(i) *= factor;
        }
    });
}

fn matmul_row_block(in_d: usize, elem: usize, rows: usize) -> usize {
    let l2 = crate::engine::runtime::l2_cache_bytes();
    if l2 == 0 || in_d == 0 || elem == 0 {
        return rows.max(1);
    }
    let budget = l2 / 2;
    let per_row = in_d.saturating_mul(elem);
    let block = (budget / per_row.max(1)).max(1);
    block.min(rows.max(1))
}

const GEMM_GPU_MIN_MACS: usize = 1 << 20;

static GEMM_GPU_STATUS_F32: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);
static GEMM_GPU_STATUS_F64: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

fn gemm_gpu_status(elem: usize) -> &'static core::sync::atomic::AtomicU8 {
    if elem == 8 { &GEMM_GPU_STATUS_F64 } else { &GEMM_GPU_STATUS_F32 }
}

fn gpu_gemm_selfcheck_f32() -> bool {
    const R: usize = 4;
    const IN: usize = 3;
    const OUT: usize = 5;
    let mut x = [0.0f32; R * IN];
    let mut w = [0.0f32; OUT * IN];
    let mut i = 0;
    while i < R * IN {
        x[i] = i as f32 * 0.5 - 1.0;
        i += 1;
    }
    i = 0;
    while i < OUT * IN {
        w[i] = i as f32 * 0.25 - 0.7;
        i += 1;
    }
    let mut y = [0.0f32; R * OUT];
    let xb = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, R * IN * 4) };
    let wb = unsafe { core::slice::from_raw_parts(w.as_ptr() as *const u8, OUT * IN * 4) };
    let yb = unsafe { core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, R * OUT * 4) };
    if !crate::engine::runtime::gpu_gemm(4, xb, yb, R, IN, OUT, wb) {
        return false;
    }
    let mut r = 0;
    while r < R {
        let mut o = 0;
        while o < OUT {
            let mut acc = 0.0f32;
            let mut k = 0;
            while k < IN {
                acc += x[r * IN + k] * w[o * IN + k];
                k += 1;
            }
            if (y[r * OUT + o] - acc).abs() > 1.0e-3 {
                return false;
            }
            o += 1;
        }
        r += 1;
    }
    true
}

fn gpu_gemm_selfcheck_f64() -> bool {
    const R: usize = 4;
    const IN: usize = 3;
    const OUT: usize = 5;
    let mut x = [0.0f64; R * IN];
    let mut w = [0.0f64; OUT * IN];
    let mut i = 0;
    while i < R * IN {
        x[i] = i as f64 * 0.5 - 1.0;
        i += 1;
    }
    i = 0;
    while i < OUT * IN {
        w[i] = i as f64 * 0.25 - 0.7;
        i += 1;
    }
    let mut y = [0.0f64; R * OUT];
    let xb = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, R * IN * 8) };
    let wb = unsafe { core::slice::from_raw_parts(w.as_ptr() as *const u8, OUT * IN * 8) };
    let yb = unsafe { core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, R * OUT * 8) };
    if !crate::engine::runtime::gpu_gemm(8, xb, yb, R, IN, OUT, wb) {
        return false;
    }
    let mut r = 0;
    while r < R {
        let mut o = 0;
        while o < OUT {
            let mut acc = 0.0f64;
            let mut k = 0;
            while k < IN {
                acc += x[r * IN + k] * w[o * IN + k];
                k += 1;
            }
            if (y[r * OUT + o] - acc).abs() > 1.0e-6 {
                return false;
            }
            o += 1;
        }
        r += 1;
    }
    true
}

fn gpu_gemm_verified(elem: usize) -> bool {
    let cell = gemm_gpu_status(elem);
    match cell.load(core::sync::atomic::Ordering::Acquire) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let ok = if elem == 8 { gpu_gemm_selfcheck_f64() } else { gpu_gemm_selfcheck_f32() };
    cell.store(if ok { 1 } else { 2 }, core::sync::atomic::Ordering::Release);
    ok
}

fn maybe_gpu_gemm<T: Float>(x: &[T], w: &[T], rows: usize, in_d: usize, out_d: usize, y: &mut [T]) -> bool {
    let elem = core::mem::size_of::<T>();
    if elem != 4 && elem != 8 {
        return false;
    }
    if rows.saturating_mul(in_d).saturating_mul(out_d) < GEMM_GPU_MIN_MACS {
        return false;
    }
    if crate::engine::get_compute_backend() != crate::engine::ComputeBackend::Gpu {
        return false;
    }
    if x.len() < rows * in_d || w.len() < out_d * in_d || y.len() < rows * out_d {
        return false;
    }
    if !gpu_gemm_verified(elem) {
        return false;
    }
    let xb = unsafe { core::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * elem) };
    let wb = unsafe { core::slice::from_raw_parts(w.as_ptr() as *const u8, w.len() * elem) };
    let yb = unsafe { core::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, y.len() * elem) };
    crate::engine::runtime::gpu_gemm(elem, xb, yb, rows, in_d, out_d, wb)
}

fn matmul_rows<T: Float>(x: &[T], w: &[T], rows: usize, in_d: usize, out_d: usize, y: &mut [T]) {
    if maybe_gpu_gemm(x, w, rows, in_d, out_d, y) {
        return;
    }
    let y_ptr = SyncMutPtr(y.as_mut_ptr());
    let block = matmul_row_block(in_d, core::mem::size_of::<T>(), rows);
    let mut rb = 0usize;
    while rb < rows {
        let rend = (rb + block).min(rows);
        parallel_for(out_d, in_d.saturating_mul(rend - rb), &move |o| {
            let _ = &y_ptr;
            let row = &w[o * in_d..o * in_d + in_d];
            let mut r = rb;
            while r < rend {
                let xr = &x[r * in_d..r * in_d + in_d];
                let acc = dot(xr, row, in_d);
                unsafe {
                    *y_ptr.0.add(r * out_d + o) = acc;
                }
                r += 1;
            }
        });
        rb = rend;
    }
}

fn matmul_rows_t_accum<T: Float>(dy: &[T], w: &[T], rows: usize, in_d: usize, out_d: usize, dx: &mut [T]) {
    let dx_ptr = SyncMutPtr(dx.as_mut_ptr());
    parallel_for(rows, out_d.saturating_mul(in_d), &move |r| {
        let _ = &dx_ptr;
        let dyr = &dy[r * out_d..r * out_d + out_d];
        let dxr = unsafe { core::slice::from_raw_parts_mut(dx_ptr.0.add(r * in_d), in_d) };
        let mut o = 0usize;
        while o < out_d {
            let a = dyr[o];
            if a.abs() > T::ZERO {
                let wo = &w[o * in_d..o * in_d + in_d];
                axpy(dxr, a, wo, in_d);
            }
            o += 1;
        }
    });
}

fn outer_rows_accum<T: Float>(dy: &[T], x: &[T], rows: usize, in_d: usize, out_d: usize, dw: &mut [T]) {
    let dw_ptr = SyncMutPtr(dw.as_mut_ptr());
    parallel_for(out_d, in_d.saturating_mul(rows), &move |o| {
        let _ = &dw_ptr;
        let dwo = unsafe { core::slice::from_raw_parts_mut(dw_ptr.0.add(o * in_d), in_d) };
        let mut r = 0usize;
        while r < rows {
            let dyo = dy[r * out_d + o];
            if dyo.abs() > T::ZERO {
                let xr = &x[r * in_d..r * in_d + in_d];
                axpy(dwo, dyo, xr, in_d);
            }
            r += 1;
        }
    });
}

#[inline]
fn silu<T: Float>(x: T) -> T {
    x / (T::ONE + (-x).exp())
}

struct Carver<'a, T> {
    buf: &'a mut [T],
}

impl<'a, T> Carver<'a, T> {
    fn take(&mut self, n: usize) -> &'a mut [T] {
        let buf = core::mem::take(&mut self.buf);
        let (head, tail) = buf.split_at_mut(n);
        self.buf = tail;
        head
    }
}

struct Work<'a, T> {
    pna: &'a mut [T],
    pota: &'a mut [T],
    q: &'a mut [T],
    k: &'a mut [T],
    v: &'a mut [T],
    probs: &'a mut [T],
    prewo: &'a mut [T],
    pnf: &'a mut [T],
    potf: &'a mut [T],
    gate: &'a mut [T],
    up: &'a mut [T],
    hidden_seq: &'a mut [T],
    hidden_in: &'a mut [T],
    final_pre: &'a mut [T],
    final_hidden: &'a mut [T],
    d_final: &'a mut [T],
    d_hidden: &'a mut [T],
    d_pota: &'a mut [T],
    d_attn_head: &'a mut [T],
    d_q: &'a mut [T],
    d_k: &'a mut [T],
    d_v: &'a mut [T],
    work_h: &'a mut [T],
    ffw_a: &'a mut [T],
    ffw_b: &'a mut [T],
    ffw_c: &'a mut [T],
    dscores: &'a mut [T],
    logits: &'a mut [T],
    probs_v: &'a mut [T],
}


pub fn batched_work_count<T: Float>(cfg: &LmConfig<T>) -> usize {
    let s = cfg.context_len;
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    let nh = cfg.num_heads;
    let v = cfg.vocab_size;
    let sh = s * h;
    let skv = s * kv_h;
    let sf = s * ffw;
    let stored = 6 * sh + 2 * skv + 2 * sf + nh * s * s;
    let working = 10 * sh + 2 * skv + 3 * sf + nh * s + s * v + v;
    stored + working
}

pub fn checkpoint_count<T: Float>(cfg: &LmConfig<T>) -> usize {
    cfg.num_layers * cfg.context_len * cfg.hidden_size
}

fn carve<'a, T: Float>(buf: &'a mut [T], cfg: &LmConfig<T>, s: usize) -> Work<'a, T> {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    let nh = cfg.num_heads;
    let v = cfg.vocab_size;
    let sh = s * h;
    let skv = s * kv_h;
    let sf = s * ffw;
    let mut c = Carver { buf };
    Work {
        pna: c.take(sh),
        pota: c.take(sh),
        q: c.take(sh),
        k: c.take(skv),
        v: c.take(skv),
        probs: c.take(nh * s * s),
        prewo: c.take(sh),
        pnf: c.take(sh),
        potf: c.take(sh),
        gate: c.take(sf),
        up: c.take(sf),
        hidden_seq: c.take(sh),
        hidden_in: c.take(sh),
        final_pre: c.take(sh),
        final_hidden: c.take(sh),
        d_final: c.take(sh),
        d_hidden: c.take(sh),
        d_pota: c.take(sh),
        d_attn_head: c.take(sh),
        d_q: c.take(sh),
        d_k: c.take(skv),
        d_v: c.take(skv),
        work_h: c.take(sh),
        ffw_a: c.take(sf),
        ffw_b: c.take(sf),
        ffw_c: c.take(sf),
        dscores: c.take(nh * s),
        logits: c.take(s * v),
        probs_v: c.take(v),
    }
}

pub fn block_param_count<T: Float>(cfg: &LmConfig<T>) -> usize {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    h + h * h + kv_h * h + kv_h * h + h * h + h + ffw * h + ffw * h + h * ffw
}

fn carve_block_w<'a, T: Float>(buf: &'a [T], cfg: &LmConfig<T>) -> LmBlockWeights<'a, T> {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    let (attn_norm_gamma, r) = buf.split_at(h);
    let (wq, r) = r.split_at(h * h);
    let (wk, r) = r.split_at(kv_h * h);
    let (wv, r) = r.split_at(kv_h * h);
    let (wo, r) = r.split_at(h * h);
    let (ffn_norm_gamma, r) = r.split_at(h);
    let (w_gate, r) = r.split_at(ffw * h);
    let (w_up, r) = r.split_at(ffw * h);
    let (w_down, _) = r.split_at(h * ffw);
    LmBlockWeights {
        attn_norm_gamma,
        wq,
        wk,
        wv,
        wo,
        ffn_norm_gamma,
        w_gate,
        w_up,
        w_down,
        moe: None,
    }
}

fn carve_block_g<'a, T: Float>(buf: &'a mut [T], cfg: &LmConfig<T>) -> LmBlockGrads<'a, T> {
    let h = cfg.hidden_size;
    let kv_h = cfg.kv_h();
    let ffw = cfg.ffw_size;
    let (attn_norm_gamma, r) = buf.split_at_mut(h);
    let (wq, r) = r.split_at_mut(h * h);
    let (wk, r) = r.split_at_mut(kv_h * h);
    let (wv, r) = r.split_at_mut(kv_h * h);
    let (wo, r) = r.split_at_mut(h * h);
    let (ffn_norm_gamma, r) = r.split_at_mut(h);
    let (w_gate, r) = r.split_at_mut(ffw * h);
    let (w_up, r) = r.split_at_mut(ffw * h);
    let (w_down, _) = r.split_at_mut(h * ffw);
    LmBlockGrads {
        attn_norm_gamma,
        wq,
        wk,
        wv,
        wo,
        ffn_norm_gamma,
        w_gate,
        w_up,
        w_down,
    }
}

fn forward_layer<T: Float>(
    cfg: &LmConfig<T>,
    bw: &LmBlockWeights<T>,
    wk: &mut Work<T>,
    s: usize,
    ops: &mut crate::observability::profiler::OpCounter,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let ffw = cfg.ffw_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let kv_h = cfg.kv_h();
    let head_dim = cfg.head_dim()?;
    let groups = cfg.kv_head_groups();
    let scale = T::ONE / T::from_usize(head_dim).sqrt();
    let eps = T::from_f32(1e-5);

    let pna = &mut wk.pna[..s * h];
    for p in 0..s {
        wk.hidden_in[p * h..p * h + h].copy_from_slice(&wk.hidden_seq[p * h..p * h + h]);
    }
    pna.copy_from_slice(&wk.hidden_seq[..s * h]);

    let pota = &mut wk.pota[..s * h];
    for p in 0..s {
        pota[p * h..p * h + h].copy_from_slice(&pna[p * h..p * h + h]);
        crate::graph::blocks::normalization::rms_norm_in_place(&mut pota[p * h..p * h + h], bw.attn_norm_gamma, eps)
            .map_err(|_| LmError::ShapeMismatch)?;
    }
    ops.add_normalization(s * h);

    let q = &mut wk.q[..s * h];
    let k = &mut wk.k[..s * kv_h];
    let vv = &mut wk.v[..s * kv_h];
    matmul_rows(pota, bw.wq, s, h, h, q);
    matmul_rows(pota, bw.wk, s, h, kv_h, k);
    matmul_rows(pota, bw.wv, s, h, kv_h, vv);
    ops.add_matmul(s, h, h);
    ops.add_matmul(s, kv_h, h);
    ops.add_matmul(s, kv_h, h);

    for p in 0..s {
        for head in 0..nh {
            let off = p * h + head * head_dim;
            crate::graph::attn::rope::apply_rope_in_place(&mut q[off..off + head_dim], p, cfg.rope_theta)
                .map_err(|_| LmError::ShapeMismatch)?;
        }
        for kvh in 0..nkv {
            let off = p * kv_h + kvh * head_dim;
            crate::graph::attn::rope::apply_rope_in_place(&mut k[off..off + head_dim], p, cfg.rope_theta)
                .map_err(|_| LmError::ShapeMismatch)?;
        }
    }

    let probs = &mut wk.probs[..nh * s * s];
    let prewo = &mut wk.prewo[..s * h];
    let probs_ptr = SyncMutPtr(probs.as_mut_ptr());
    let prewo_ptr = SyncMutPtr(prewo.as_mut_ptr());
    let q_ro: &[T] = q;
    let k_ro: &[T] = k;
    let v_ro: &[T] = vv;
    parallel_for(nh, s * head_dim, &move |head| {
        let _ = (&probs_ptr, &prewo_ptr);
        let kv_head = head / groups;
        let kv_off = kv_head * head_dim;
        let q_head = head * head_dim;
        for p in 0..s {
            let qbase = p * h + q_head;
            let prow = head * s * s + p * s;
            let seq = p + 1;
            let mut max_s = T::NEG_INF;
            for t in 0..seq {
                let kbase = t * kv_h + kv_off;
                let mut acc = T::ZERO;
                for i in 0..head_dim {
                    acc += q_ro[qbase + i] * k_ro[kbase + i];
                }
                let sc = acc * scale;
                unsafe {
                    *probs_ptr.0.add(prow + t) = sc;
                }
                if sc > max_s {
                    max_s = sc;
                }
            }
            let mut sum = T::ZERO;
            for t in 0..seq {
                let e = (unsafe { *probs_ptr.0.add(prow + t) } - max_s).exp();
                unsafe {
                    *probs_ptr.0.add(prow + t) = e;
                }
                sum += e;
            }
            let inv = if sum > T::ZERO { T::ONE / sum } else { T::ZERO };
            let obase = p * h + q_head;
            for i in 0..head_dim {
                unsafe {
                    *prewo_ptr.0.add(obase + i) = T::ZERO;
                }
            }
            for t in 0..seq {
                let pw = unsafe { *probs_ptr.0.add(prow + t) } * inv;
                unsafe {
                    *probs_ptr.0.add(prow + t) = pw;
                }
                let vbase = t * kv_h + kv_off;
                for i in 0..head_dim {
                    unsafe {
                        *prewo_ptr.0.add(obase + i) += pw * v_ro[vbase + i];
                    }
                }
            }
        }
    });
    for _ in 0..nh {
        ops.add_attention(s, s, head_dim);
    }

    matmul_rows(prewo, bw.wo, s, h, h, wk.work_h);
    ops.add_matmul(s, h, h);
    let pnf = &mut wk.pnf[..s * h];
    for idx in 0..s * h {
        wk.hidden_seq[idx] += wk.work_h[idx];
    }
    pnf.copy_from_slice(&wk.hidden_seq[..s * h]);

    let potf = &mut wk.potf[..s * h];
    for p in 0..s {
        potf[p * h..p * h + h].copy_from_slice(&pnf[p * h..p * h + h]);
        crate::graph::blocks::normalization::rms_norm_in_place(&mut potf[p * h..p * h + h], bw.ffn_norm_gamma, eps)
            .map_err(|_| LmError::ShapeMismatch)?;
    }
    ops.add_normalization(s * h);

    let gate = &mut wk.gate[..s * ffw];
    let up = &mut wk.up[..s * ffw];
    matmul_rows(potf, bw.w_gate, s, h, ffw, gate);
    matmul_rows(potf, bw.w_up, s, h, ffw, up);
    ops.add_matmul(s, ffw, h);
    ops.add_matmul(s, ffw, h);

    for idx in 0..s * ffw {
        wk.ffw_a[idx] = silu(gate[idx]) * up[idx];
    }
    ops.add_activation(s * ffw);
    matmul_rows(wk.ffw_a, bw.w_down, s, ffw, h, wk.work_h);
    ops.add_matmul(s, h, ffw);
    for idx in 0..s * h {
        wk.hidden_seq[idx] += wk.work_h[idx];
    }
    Ok(())
}

fn backward_layer<T: Float>(
    cfg: &LmConfig<T>,
    bw: &LmBlockWeights<T>,
    g: &mut LmBlockGrads<T>,
    wk: &mut Work<T>,
    s: usize,
) -> Result<(), LmError> {
    let h = cfg.hidden_size;
    let ffw = cfg.ffw_size;
    let nkv = cfg.num_kv_heads;
    let kv_h = cfg.kv_h();
    let head_dim = cfg.head_dim()?;
    let groups = cfg.kv_head_groups();
    let scale = T::ONE / T::from_usize(head_dim).sqrt();

    let pna = &wk.pna[..s * h];
    let pota = &wk.pota[..s * h];
    let q = &wk.q[..s * h];
    let k = &wk.k[..s * kv_h];
    let vv = &wk.v[..s * kv_h];
    let probs = &wk.probs[..nkv * groups * s * s];
    let prewo = &wk.prewo[..s * h];
    let pnf = &wk.pnf[..s * h];
    let potf = &wk.potf[..s * h];
    let gate = &wk.gate[..s * ffw];
    let up = &wk.up[..s * ffw];

    for idx in 0..s * ffw {
        wk.ffw_a[idx] = silu(gate[idx]) * up[idx];
    }
    outer_rows_accum(&wk.d_hidden[..s * h], wk.ffw_a, s, ffw, h, g.w_down);

    for g2 in wk.ffw_b[..s * ffw].iter_mut() {
        *g2 = T::ZERO;
    }
    matmul_rows_t_accum(&wk.d_hidden[..s * h], bw.w_down, s, ffw, h, wk.ffw_b);

    for idx in 0..s * ffw {
        let d_gs = wk.ffw_b[idx];
        let sp = crate::engine::infer::inference::transformer_block::silu_prime(gate[idx]);
        let sg = silu(gate[idx]);
        wk.ffw_a[idx] = d_gs * sp * up[idx];
        wk.ffw_c[idx] = d_gs * sg;
    }
    outer_rows_accum(wk.ffw_a, potf, s, h, ffw, g.w_gate);
    outer_rows_accum(wk.ffw_c, potf, s, h, ffw, g.w_up);

    for d in wk.d_pota[..s * h].iter_mut() {
        *d = T::ZERO;
    }
    matmul_rows_t_accum(wk.ffw_a, bw.w_gate, s, h, ffw, wk.d_pota);
    matmul_rows_t_accum(wk.ffw_c, bw.w_up, s, h, ffw, wk.d_pota);

    for p in 0..s {
        for d in wk.work_h[p * h..p * h + h].iter_mut() {
            *d = T::ZERO;
        }
        crate::engine::infer::inference::rmsnorm_backward(
            &pnf[p * h..p * h + h],
            bw.ffn_norm_gamma,
            &wk.d_pota[p * h..p * h + h],
            &mut g.ffn_norm_gamma[..h],
            &mut wk.work_h[p * h..p * h + h],
            h,
        );
        for (d, &t) in wk.d_hidden[p * h..p * h + h].iter_mut().zip(wk.work_h[p * h..p * h + h].iter()) {
            *d += t;
        }
    }

    outer_rows_accum(&wk.d_hidden[..s * h], prewo, s, h, h, g.wo);
    for d in wk.d_attn_head[..s * h].iter_mut() {
        *d = T::ZERO;
    }
    matmul_rows_t_accum(&wk.d_hidden[..s * h], bw.wo, s, h, h, wk.d_attn_head);

    for d in wk.d_q[..s * h].iter_mut() {
        *d = T::ZERO;
    }
    for d in wk.d_k[..s * kv_h].iter_mut() {
        *d = T::ZERO;
    }
    for d in wk.d_v[..s * kv_h].iter_mut() {
        *d = T::ZERO;
    }

    let dscores_ptr = SyncMutPtr(wk.dscores.as_mut_ptr());
    let dq_ptr = SyncMutPtr(wk.d_q.as_mut_ptr());
    let dk_ptr = SyncMutPtr(wk.d_k.as_mut_ptr());
    let dv_ptr = SyncMutPtr(wk.d_v.as_mut_ptr());
    let d_attn: &[T] = wk.d_attn_head;
    parallel_for(nkv, (groups * s) * head_dim, &move |kv_head| {
        let _ = (&dscores_ptr, &dq_ptr, &dk_ptr, &dv_ptr);
        let kv_off = kv_head * head_dim;
        for hg in 0..groups {
            let head = kv_head * groups + hg;
            let q_head = head * head_dim;
            let drow = head * s;
            for p in 0..s {
                let seq = p + 1;
                let prow = head * s * s + p * s;
                let abase = p * h + q_head;
                let mut dot_s_ds = T::ZERO;
                for t in 0..seq {
                    let vbase = t * kv_h + kv_off;
                    let mut d_st = T::ZERO;
                    for i in 0..head_dim {
                        d_st += d_attn[abase + i] * vv[vbase + i];
                    }
                    unsafe {
                        *dscores_ptr.0.add(drow + t) = d_st;
                    }
                    dot_s_ds += probs[prow + t] * d_st;
                }
                for t in 0..seq {
                    let ds = probs[prow + t] * (unsafe { *dscores_ptr.0.add(drow + t) } - dot_s_ds) * scale;
                    unsafe {
                        *dscores_ptr.0.add(drow + t) = ds;
                    }
                }
                for t in 0..seq {
                    let ds = unsafe { *dscores_ptr.0.add(drow + t) };
                    let kbase = t * kv_h + kv_off;
                    for i in 0..head_dim {
                        unsafe {
                            *dq_ptr.0.add(abase + i) += ds * k[kbase + i];
                        }
                    }
                }
                let ds_diag = unsafe { *dscores_ptr.0.add(drow + p) };
                let dkbase = p * kv_h + kv_off;
                for i in 0..head_dim {
                    unsafe {
                        *dk_ptr.0.add(dkbase + i) += ds_diag * q[abase + i];
                    }
                }
                let s_diag = probs[prow + p];
                for i in 0..head_dim {
                    unsafe {
                        *dv_ptr.0.add(dkbase + i) += s_diag * d_attn[abase + i];
                    }
                }
            }
        }
    });

    outer_rows_accum(&wk.d_q[..s * h], pota, s, h, h, g.wq);
    outer_rows_accum(&wk.d_k[..s * kv_h], pota, s, h, kv_h, g.wk);
    outer_rows_accum(&wk.d_v[..s * kv_h], pota, s, h, kv_h, g.wv);

    for d in wk.d_pota[..s * h].iter_mut() {
        *d = T::ZERO;
    }
    matmul_rows_t_accum(&wk.d_q[..s * h], bw.wq, s, h, h, wk.d_pota);
    matmul_rows_t_accum(&wk.d_k[..s * kv_h], bw.wk, s, h, kv_h, wk.d_pota);

    for p in 0..s {
        for d in wk.work_h[p * h..p * h + h].iter_mut() {
            *d = T::ZERO;
        }
        crate::engine::infer::inference::rmsnorm_backward(
            &pna[p * h..p * h + h],
            bw.attn_norm_gamma,
            &wk.d_pota[p * h..p * h + h],
            &mut g.attn_norm_gamma[..h],
            &mut wk.work_h[p * h..p * h + h],
            h,
        );
        for (d, &t) in wk.d_hidden[p * h..p * h + h].iter_mut().zip(wk.work_h[p * h..p * h + h].iter()) {
            *d += t;
        }
    }
    Ok(())
}

pub struct TrainBatchedBuffers<'a, T> {
    pub params: &'a mut [T],
    pub m: &'a mut [T],
    pub v: &'a mut [T],
    pub grad_layer: &'a mut [T],
    pub embed_grad: &'a mut [T],
    pub final_grad: &'a mut [T],
    pub work: &'a mut [T],
    pub checkpoints: &'a mut [T],
}

pub fn forward_loss_batched<T: Float>(
    cfg: &LmConfig<T>,
    params: &[T],
    token_ids: &[u32],
    work: &mut [T],
) -> Result<T, LmError> {
    if token_ids.len() < 2 {
        return Err(LmError::ShapeMismatch);
    }
    let s = token_ids.len() - 1;
    if s > cfg.context_len {
        return Err(LmError::ShapeMismatch);
    }
    if work.len() < batched_work_count(cfg) {
        return Err(LmError::ScratchTooSmall);
    }

    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let nl = cfg.num_layers;
    let bpc = block_param_count(cfg);
    let sh = s * h;
    let final_off = v * h + nl * bpc;
    let n_params = v * h + nl * bpc + h;
    if params.len() < n_params {
        return Err(LmError::ShapeMismatch);
    }

    let eps = T::from_f32(1e-5);
    let mut wk = carve(work, cfg, s);

    for (p, &token_id) in token_ids.iter().take(s).enumerate() {
        crate::graph::blocks::embeddings::gather_embeddings(
            &params[0..v * h],
            v,
            h,
            &[token_id as usize],
            &mut wk.hidden_seq[p * h..p * h + h],
        )
        .map_err(|_| LmError::ShapeMismatch)?;
    }

    let mut prof_ops = crate::observability::profiler::OpCounter::new();
    for layer in 0..nl {
        let off = v * h + layer * bpc;
        let bw = carve_block_w(&params[off..off + bpc], cfg);
        forward_layer(cfg, &bw, &mut wk, s, &mut prof_ops)?;
    }

    wk.final_pre[..sh].copy_from_slice(&wk.hidden_seq[..sh]);
    let final_gamma = &params[final_off..final_off + h];
    for p in 0..s {
        wk.final_hidden[p * h..p * h + h].copy_from_slice(&wk.final_pre[p * h..p * h + h]);
        crate::graph::blocks::normalization::rms_norm_in_place(
            &mut wk.final_hidden[p * h..p * h + h],
            final_gamma,
            eps,
        )
        .map_err(|_| LmError::ShapeMismatch)?;
    }

    matmul_rows(&wk.final_hidden[..sh], &params[0..v * h], s, h, v, wk.logits);

    let mut total = T::ZERO;
    for (p, &target_token) in token_ids.iter().skip(1).take(s).enumerate() {
        let lp = &mut wk.logits[p * v..p * v + v];
        let target = target_token as usize;
        if target >= v {
            return Err(LmError::NonFinite);
        }
        let loss = cross_entropy(lp, target, wk.probs_v).map_err(|_| LmError::NonFinite)?;
        total += loss;
    }

    Ok(total / T::from_usize(s))
}

#[cfg(feature = "publisher-trust-service")]
pub fn forward_predict_batched<T: Float>(
    cfg: &LmConfig<T>,
    params: &[T],
    token_ids: &[u32],
    work: &mut [T],
    out: &mut [u32],
) -> Result<usize, LmError> {
    if token_ids.len() < 2 {
        return Err(LmError::ShapeMismatch);
    }
    let s = token_ids.len() - 1;
    if s > cfg.context_len {
        return Err(LmError::ShapeMismatch);
    }
    if work.len() < batched_work_count(cfg) {
        return Err(LmError::ScratchTooSmall);
    }
    if out.len() < s {
        return Err(LmError::ShapeMismatch);
    }

    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let nl = cfg.num_layers;
    let bpc = block_param_count(cfg);
    let sh = s * h;
    let final_off = v * h + nl * bpc;
    let n_params = v * h + nl * bpc + h;
    if params.len() < n_params {
        return Err(LmError::ShapeMismatch);
    }

    let eps = T::from_f32(1e-5);
    let mut wk = carve(work, cfg, s);

    for (p, &token_id) in token_ids.iter().take(s).enumerate() {
        crate::graph::blocks::embeddings::gather_embeddings(
            &params[0..v * h],
            v,
            h,
            &[token_id as usize],
            &mut wk.hidden_seq[p * h..p * h + h],
        )
        .map_err(|_| LmError::ShapeMismatch)?;
    }

    let mut prof_ops = crate::observability::profiler::OpCounter::new();
    for layer in 0..nl {
        let off = v * h + layer * bpc;
        let bw = carve_block_w(&params[off..off + bpc], cfg);
        forward_layer(cfg, &bw, &mut wk, s, &mut prof_ops)?;
    }

    wk.final_pre[..sh].copy_from_slice(&wk.hidden_seq[..sh]);
    let final_gamma = &params[final_off..final_off + h];
    for p in 0..s {
        wk.final_hidden[p * h..p * h + h].copy_from_slice(&wk.final_pre[p * h..p * h + h]);
        crate::graph::blocks::normalization::rms_norm_in_place(
            &mut wk.final_hidden[p * h..p * h + h],
            final_gamma,
            eps,
        )
        .map_err(|_| LmError::ShapeMismatch)?;
    }

    matmul_rows(&wk.final_hidden[..sh], &params[0..v * h], s, h, v, wk.logits);

    for (p, o) in out.iter_mut().enumerate().take(s) {
        let lp = &wk.logits[p * v..p * v + v];
        let mut best = 0usize;
        let mut best_val = lp[0];
        for (k, &val) in lp.iter().enumerate().take(v).skip(1) {
            if val > best_val {
                best_val = val;
                best = k;
            }
        }
        *o = best as u32;
    }

    Ok(s)
}

pub fn train_batched<T: Float>(
    cfg: &LmConfig<T>,
    train_cfg: &LmTrainConfig<T>,
    bufs: &mut TrainBatchedBuffers<T>,
    token_ids: &[u32],
    target_active: Option<&[bool]>,
    target_override: Option<&[u32]>,
    step: u32,
) -> Result<LmTrainStep<T>, LmError> {
    if token_ids.len() < 2 {
        return Err(LmError::ShapeMismatch);
    }
    let s = token_ids.len() - 1;
    if s > cfg.context_len {
        return Err(LmError::ShapeMismatch);
    }
    if bufs.work.len() < batched_work_count(cfg) {
        return Err(LmError::ScratchTooSmall);
    }

    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let nl = cfg.num_layers;
    let bpc = block_param_count(cfg);
    let sh = s * h;
    let n_params = v * h + nl * bpc + h;
    let final_off = v * h + nl * bpc;

    if bufs.params.len() < n_params || bufs.m.len() < n_params || bufs.v.len() < n_params {
        return Err(LmError::ShapeMismatch);
    }
    if bufs.grad_layer.len() < bpc {
        return Err(LmError::ShapeMismatch);
    }
    if bufs.embed_grad.len() < v * h || bufs.final_grad.len() < h {
        return Err(LmError::ShapeMismatch);
    }
    if bufs.checkpoints.len() < nl * cfg.context_len * h {
        return Err(LmError::ShapeMismatch);
    }

    let eps = T::from_f32(1e-5);
    let eps_adam = T::from_f32(1e-8);

    fill_zero(&mut bufs.embed_grad[..v * h]);
    for x in bufs.final_grad[..h].iter_mut() {
        *x = T::ZERO;
    }

    let mut wk = carve(bufs.work, cfg, s);

    for (p, &token_id) in token_ids.iter().take(s).enumerate() {
        crate::graph::blocks::embeddings::gather_embeddings(
            &bufs.params[0..v * h],
            v,
            h,
            &[token_id as usize],
            &mut wk.hidden_seq[p * h..p * h + h],
        )
        .map_err(|_| LmError::ShapeMismatch)?;
    }

    let mut prof_ops = crate::observability::profiler::OpCounter::new();
    for layer in 0..nl {
        bufs.checkpoints[layer * sh..layer * sh + sh].copy_from_slice(&wk.hidden_seq[..sh]);
        let off = v * h + layer * bpc;
        let bw = carve_block_w(&bufs.params[off..off + bpc], cfg);
        forward_layer(cfg, &bw, &mut wk, s, &mut prof_ops)?;
    }

    wk.final_pre[..sh].copy_from_slice(&wk.hidden_seq[..sh]);
    {
        let final_gamma = &bufs.params[final_off..final_off + h];
        for p in 0..s {
            wk.final_hidden[p * h..p * h + h].copy_from_slice(&wk.final_pre[p * h..p * h + h]);
            crate::graph::blocks::normalization::rms_norm_in_place(
                &mut wk.final_hidden[p * h..p * h + h],
                final_gamma,
                eps,
            )
            .map_err(|_| LmError::ShapeMismatch)?;
        }
    }

    let mut total_loss = T::ZERO;
    let mut active_count = 0usize;
    for gg in wk.d_final[..sh].iter_mut() {
        *gg = T::ZERO;
    }
    matmul_rows(&wk.final_hidden[..sh], &bufs.params[0..v * h], s, h, v, wk.logits);
    prof_ops.add_normalization(s * h);
    prof_ops.add_matmul(s, v, h);
    for p in 0..s {
        let is_active = match target_active {
            Some(mask) => mask[p],
            None => true,
        };
        let lp = &mut wk.logits[p * v..p * v + v];
        if !is_active {
            for x in lp.iter_mut() {
                *x = T::ZERO;
            }
            continue;
        }
        let target = token_ids[p + 1] as usize;
        if target >= v {
            return Err(LmError::NonFinite);
        }
        let target = match target_override {
            Some(ov) if p < ov.len() => ov[p] as usize,
            _ => target,
        };
        if target >= v {
            return Err(LmError::NonFinite);
        }
        let loss = cross_entropy(lp, target, wk.probs_v).map_err(|_| LmError::NonFinite)?;
        total_loss += loss;
        active_count += 1;
        lp.copy_from_slice(&wk.probs_v[..v]);
        lp[target] -= T::ONE;
    }

    let denom = if target_active.is_some() { active_count } else { s };
    let lr = compute_learning_rate(train_cfg.base_lr, step, train_cfg.schedule)
        .unwrap_or(train_cfg.base_lr);
    if denom == 0 {
        return Ok(LmTrainStep { loss: T::ZERO, grad_norm: T::ZERO, lr, ops: prof_ops });
    }
    let inv_n = T::ONE / T::from_usize(denom);

    outer_rows_accum(&wk.logits[..s * v], &wk.final_hidden[..sh], s, h, v, &mut bufs.embed_grad[..v * h]);
    matmul_rows_t_accum(&wk.logits[..s * v], &bufs.params[0..v * h], s, v, h, &mut wk.d_final[..sh]);

    {
        let final_gamma = &bufs.params[final_off..final_off + h];
        for p in 0..s {
            for gg in wk.d_hidden[p * h..p * h + h].iter_mut() {
                *gg = T::ZERO;
            }
            crate::engine::infer::inference::rmsnorm_backward(
                &wk.final_pre[p * h..p * h + h],
                final_gamma,
                &wk.d_final[p * h..p * h + h],
                &mut bufs.final_grad[..h],
                &mut wk.d_hidden[p * h..p * h + h],
                h,
            );
        }
    }

    let mut section_norms = [T::ZERO; MAX_LAYERS + 2];
    let mut nsec = 0usize;
    for layer in (0..nl).rev() {
        let off = v * h + layer * bpc;
        wk.hidden_seq[..sh].copy_from_slice(&bufs.checkpoints[layer * sh..layer * sh + sh]);
        for x in bufs.grad_layer[..bpc].iter_mut() {
            *x = T::ZERO;
        }
        {
            let bw = carve_block_w(&bufs.params[off..off + bpc], cfg);
            forward_layer(cfg, &bw, &mut wk, s, &mut prof_ops)?;
            let mut g = carve_block_g(&mut bufs.grad_layer[..bpc], cfg);
            backward_layer(cfg, &bw, &mut g, &mut wk, s)?;
        }
        for x in bufs.grad_layer[..bpc].iter_mut() {
            *x *= inv_n;
        }
        let layer_norm = if train_cfg.grad_clip > T::ZERO {
            clip_by_global_norm(&mut bufs.grad_layer[..bpc], train_cfg.grad_clip).unwrap_or(T::ZERO)
        } else {
            T::ZERO
        };
        section_norms[nsec] = layer_norm;
        nsec += 1;
        step_adamw(
            &mut bufs.params[off..off + bpc],
            &bufs.grad_layer[..bpc],
            &mut bufs.m[off..off + bpc],
            &mut bufs.v[off..off + bpc],
            AdamwConfig {
                learning_rate: lr,
                step,
                beta1: train_cfg.beta1,
                beta2: train_cfg.beta2,
                eps: eps_adam,
                weight_decay: train_cfg.weight_decay,
            },
        )
        .map_err(map_opt_err)?;
    }

    for p in 0..s {
        let is_active = match target_active {
            Some(mask) => mask[p],
            None => true,
        };
        if !is_active {
            continue;
        }
        let tok = token_ids[p] as usize;
        let goff = tok * h;
        if goff + h <= bufs.embed_grad.len() {
            for (d, &gv) in bufs.embed_grad[goff..goff + h]
                .iter_mut()
                .zip(wk.d_hidden[p * h..p * h + h].iter())
            {
                *d += gv;
            }
        }
    }

    scale_slice(&mut bufs.embed_grad[..v * h], inv_n);
    for x in bufs.final_grad[..h].iter_mut() {
        *x *= inv_n;
    }

    let embed_norm = if train_cfg.grad_clip > T::ZERO {
        clip_by_global_norm(&mut bufs.embed_grad[..v * h], train_cfg.grad_clip).unwrap_or(T::ZERO)
    } else {
        T::ZERO
    };
    section_norms[nsec] = embed_norm;
    nsec += 1;
    step_adamw(
        &mut bufs.params[0..v * h],
        &bufs.embed_grad[..v * h],
        &mut bufs.m[0..v * h],
        &mut bufs.v[0..v * h],
        AdamwConfig {
            learning_rate: lr,
            step,
            beta1: train_cfg.beta1,
            beta2: train_cfg.beta2,
            eps: eps_adam,
            weight_decay: train_cfg.weight_decay,
        },
    )
    .map_err(map_opt_err)?;

    let final_norm = if train_cfg.grad_clip > T::ZERO {
        clip_by_global_norm(&mut bufs.final_grad[..h], train_cfg.grad_clip).unwrap_or(T::ZERO)
    } else {
        T::ZERO
    };
    section_norms[nsec] = final_norm;
    nsec += 1;
    step_adamw(
        &mut bufs.params[final_off..final_off + h],
        &bufs.final_grad[..h],
        &mut bufs.m[final_off..final_off + h],
        &mut bufs.v[final_off..final_off + h],
        AdamwConfig {
            learning_rate: lr,
            step,
            beta1: train_cfg.beta1,
            beta2: train_cfg.beta2,
            eps: eps_adam,
            weight_decay: train_cfg.weight_decay,
        },
    )
    .map_err(map_opt_err)?;

    total_loss *= inv_n;
    Ok(LmTrainStep {
        loss: total_loss,
        grad_norm: l2_norm(&section_norms[..nsec]).unwrap_or(T::ZERO),
        lr,
        ops: prof_ops,
    })
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::graph::lm::lm_weights_param_count;
    use std::vec;
    use std::vec::Vec;

    fn tiny_cfg() -> LmConfig<f32> {
        LmConfig {
            vocab_size: 19,
            context_len: 6,
            hidden_size: 8,
            ffw_size: 16,
            num_layers: 3,
            num_heads: 2,
            num_kv_heads: 1,
            rope_theta: 10_000.0,
            head_hidden: 8,
        }
    }

    fn make_params(n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for slot in out.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = ((state >> 33) as u32) as f32 / (u32::MAX as f32);
            *slot = r - 0.5;
        }
        out
    }

    #[test]
    fn train_batched_reduces_loss() {
        let cfg = tiny_cfg();
        let train_cfg = LmTrainConfig::<f32>::default_adam();
        let n = lm_weights_param_count(&cfg);
        let v = cfg.vocab_size;
        let h = cfg.hidden_size;
        let bpc = block_param_count(&cfg);
        let token_ids: [u32; 6] = [1, 5, 3, 9, 2, 7];

        let mut params = make_params(n);
        let mut m = vec![0.0f32; n];
        let mut vbuf = vec![0.0f32; n];
        let mut grad_layer = vec![0.0f32; bpc];
        let mut embed_grad = vec![0.0f32; v * h];
        let mut final_grad = vec![0.0f32; h];
        let mut work = vec![0.0f32; batched_work_count(&cfg)];
        let mut checkpoints = vec![0.0f32; checkpoint_count(&cfg)];

        let mut first = 0.0f32;
        let mut last = 0.0f32;
        for stp in 0..60u32 {
            let mut bufs = TrainBatchedBuffers {
                params: &mut params,
                m: &mut m,
                v: &mut vbuf,
                grad_layer: &mut grad_layer,
                embed_grad: &mut embed_grad,
                final_grad: &mut final_grad,
                work: &mut work,
                checkpoints: &mut checkpoints,
            };
            let out = train_batched(&cfg, &train_cfg, &mut bufs, &token_ids, None, None, stp + 1).unwrap();
            assert!(out.loss.is_finite());
            assert!(out.grad_norm.is_finite());
            if stp == 0 {
                first = out.loss;
            }
            last = out.loss;
        }
        assert!(last < first, "loss did not decrease: first={} last={}", first, last);
    }
}
