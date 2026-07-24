use core::sync::atomic::{AtomicU8, Ordering};

static SIMD_STATE: AtomicU8 = AtomicU8::new(0);

#[cfg(target_arch = "x86_64")]
fn cpuid(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "xchg {tmp:r}, rbx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "xchg {tmp:r}, rbx",
            tmp = out(reg) _,
            ebx_out = out(reg) ebx,
            inout("eax") leaf => eax,
            inout("ecx") sub => ecx,
            out("edx") edx,
            options(nostack, preserves_flags)
        );
    }
    (eax, ebx, ecx, edx)
}

#[cfg(target_arch = "x86_64")]
fn xcr0() -> u64 {
    let eax: u32;
    let edx: u32;
    unsafe {
        core::arch::asm!(
            "xgetbv",
            in("ecx") 0u32,
            out("eax") eax,
            out("edx") edx,
            options(nostack, preserves_flags)
        );
    }
    ((edx as u64) << 32) | (eax as u64)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> bool {
    let (_, _, ecx1, _) = cpuid(1, 0);
    let fma = ecx1 & (1 << 12) != 0;
    let osxsave = ecx1 & (1 << 27) != 0;
    let avx = ecx1 & (1 << 28) != 0;
    if !(fma && osxsave && avx) {
        return false;
    }
    let state = xcr0();
    if state & 0b110 != 0b110 {
        return false;
    }
    let (_, ebx7, _, _) = cpuid(7, 0);
    ebx7 & (1 << 5) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn detect() -> bool {
    false
}

#[inline]
fn simd_enabled() -> bool {
    let cached = SIMD_STATE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 1;
    }
    let enabled = detect();
    SIMD_STATE.store(if enabled { 1 } else { 2 }, Ordering::Relaxed);
    enabled
}

fn dot_scalar(a: &[f32], b: &[f32], n: usize) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut acc2 = 0.0f32;
    let mut acc3 = 0.0f32;
    let mut acc4 = 0.0f32;
    let mut acc5 = 0.0f32;
    let mut acc6 = 0.0f32;
    let mut acc7 = 0.0f32;
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

fn axpy_scalar(dst: &mut [f32], a: f32, src: &[f32], n: usize) {
    let mut i = 0usize;
    while i + 8 <= n {
        dst[i] += a * src[i];
        dst[i + 1] += a * src[i + 1];
        dst[i + 2] += a * src[i + 2];
        dst[i + 3] += a * src[i + 3];
        dst[i + 4] += a * src[i + 4];
        dst[i + 5] += a * src[i + 5];
        dst[i + 6] += a * src[i + 6];
        dst[i + 7] += a * src[i + 7];
        i += 8;
    }
    while i < n {
        dst[i] += a * src[i];
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx(a: &[f32], b: &[f32], n: usize) -> f32 {
    use core::arch::x86_64::*;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 32 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), acc3);
        i += 32;
    }
    while i + 8 <= n {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
        i += 8;
    }
    let acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(hi, lo);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(sums2);
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx(dst: &mut [f32], a: f32, src: &[f32], n: usize) {
    use core::arch::x86_64::*;
    let va = _mm256_set1_ps(a);
    let pd = dst.as_mut_ptr();
    let ps = src.as_ptr();
    let mut i = 0usize;
    while i + 32 <= n {
        let d0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(ps.add(i)), _mm256_loadu_ps(pd.add(i)));
        _mm256_storeu_ps(pd.add(i), d0);
        let d1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(ps.add(i + 8)), _mm256_loadu_ps(pd.add(i + 8)));
        _mm256_storeu_ps(pd.add(i + 8), d1);
        let d2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(ps.add(i + 16)), _mm256_loadu_ps(pd.add(i + 16)));
        _mm256_storeu_ps(pd.add(i + 16), d2);
        let d3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(ps.add(i + 24)), _mm256_loadu_ps(pd.add(i + 24)));
        _mm256_storeu_ps(pd.add(i + 24), d3);
        i += 32;
    }
    while i + 8 <= n {
        let d0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(ps.add(i)), _mm256_loadu_ps(pd.add(i)));
        _mm256_storeu_ps(pd.add(i), d0);
        i += 8;
    }
    while i < n {
        *pd.add(i) += a * *ps.add(i);
        i += 1;
    }
}

#[inline(always)]
pub(crate) fn dot_f32(a: &[f32], b: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            return unsafe { dot_avx(a, b, n) };
        }
    }
    dot_scalar(a, b, n)
}

#[inline(always)]
pub(crate) fn axpy_f32(dst: &mut [f32], a: f32, src: &[f32], n: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if simd_enabled() {
            unsafe { axpy_avx(dst, a, src, n) };
            return;
        }
    }
    axpy_scalar(dst, a, src, n)
}
