use core::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

static AVX2_STATE: AtomicU8 = AtomicU8::new(0);
static L2_CACHE_STATE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn cpuid(leaf: u32, sub: u32) -> (u32, u32, u32, u32) {
    let eax: u32;
    let ebx: u32;
    let ecx: u32;
    let edx: u32;
    #[cfg(target_arch = "x86_64")]
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
    #[cfg(target_arch = "x86")]
    unsafe {
        core::arch::asm!(
            "xchg {tmp:e}, ebx",
            "cpuid",
            "mov {ebx_out:e}, ebx",
            "xchg {tmp:e}, ebx",
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

fn detect() -> bool {
    let (_, _, ecx1, _) = cpuid(1, 0);
    let fma = ecx1 & (1 << 12) != 0;
    let osxsave = ecx1 & (1 << 27) != 0;
    let avx = ecx1 & (1 << 28) != 0;
    if !(fma && osxsave && avx) {
        return false;
    }
    if xcr0() & 0b110 != 0b110 {
        return false;
    }
    let (_, ebx7, _, _) = cpuid(7, 0);
    ebx7 & (1 << 5) != 0
}

pub(crate) fn avx2_fma_available() -> bool {
    let cached = AVX2_STATE.load(Ordering::Relaxed);
    if cached != 0 {
        return cached == 1;
    }
    let enabled = detect();
    AVX2_STATE.store(if enabled { 1 } else { 2 }, Ordering::Relaxed);
    enabled
}

fn cache_descriptor(leaf: u32, idx: u32) -> Option<(u32, u32, usize)> {
    let (eax, ebx, ecx, _) = cpuid(leaf, idx);
    let ctype = eax & 0x1F;
    if ctype == 0 {
        return None;
    }
    let level = (eax >> 5) & 0x7;
    let ways = ((ebx >> 22) & 0x3FF) + 1;
    let parts = ((ebx >> 12) & 0x3FF) + 1;
    let line = (ebx & 0xFFF) + 1;
    let sets = ecx + 1;
    let size = ways as usize * parts as usize * line as usize * sets as usize;
    Some((level, ctype, size))
}

fn detect_l2() -> usize {
    let (_, vb, vc, vd) = cpuid(0, 0);
    let is_amd = vb == 0x6874_7541 && vd == 0x6974_6E65 && vc == 0x444D_4163;
    let (maxext, _, _, _) = cpuid(0x8000_0000, 0);
    let leaf = if is_amd && maxext >= 0x8000_001D {
        0x8000_001D
    } else {
        4
    };
    let mut best = 0usize;
    let mut idx = 0u32;
    while idx < 16 {
        match cache_descriptor(leaf, idx) {
            Some((level, ctype, size)) => {
                if level == 2 && (ctype == 1 || ctype == 3) {
                    best = size;
                }
            }
            None => break,
        }
        idx += 1;
    }
    best
}

pub(crate) fn l2_cache_bytes() -> usize {
    let cached = L2_CACHE_STATE.load(Ordering::Relaxed);
    if cached != usize::MAX {
        return cached;
    }
    let size = detect_l2();
    L2_CACHE_STATE.store(size, Ordering::Relaxed);
    size
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_avx(a: &[f32], b: &[f32], n: usize) -> f32 {
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

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn axpy_avx(dst: &mut [f32], a: f32, src: &[f32], n: usize) {
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

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_avx_f64(a: &[f64], b: &[f64], n: usize) -> f64 {
    use core::arch::x86_64::*;
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();
    let mut acc2 = _mm256_setzero_pd();
    let mut acc3 = _mm256_setzero_pd();
    let mut i = 0usize;
    while i + 16 <= n {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), _mm256_loadu_pd(pb.add(i)), acc0);
        acc1 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i + 4)), _mm256_loadu_pd(pb.add(i + 4)), acc1);
        acc2 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i + 8)), _mm256_loadu_pd(pb.add(i + 8)), acc2);
        acc3 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i + 12)), _mm256_loadu_pd(pb.add(i + 12)), acc3);
        i += 16;
    }
    while i + 4 <= n {
        acc0 = _mm256_fmadd_pd(_mm256_loadu_pd(pa.add(i)), _mm256_loadu_pd(pb.add(i)), acc0);
        i += 4;
    }
    let acc = _mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3));
    let hi = _mm256_extractf128_pd(acc, 1);
    let lo = _mm256_castpd256_pd128(acc);
    let sum128 = _mm_add_pd(hi, lo);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sums = _mm_add_sd(sum128, shuf);
    let mut total = _mm_cvtsd_f64(sums);
    while i < n {
        total += *pa.add(i) * *pb.add(i);
        i += 1;
    }
    total
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn axpy_avx_f64(dst: &mut [f64], a: f64, src: &[f64], n: usize) {
    use core::arch::x86_64::*;
    let va = _mm256_set1_pd(a);
    let pd = dst.as_mut_ptr();
    let ps = src.as_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let d0 = _mm256_fmadd_pd(va, _mm256_loadu_pd(ps.add(i)), _mm256_loadu_pd(pd.add(i)));
        _mm256_storeu_pd(pd.add(i), d0);
        let d1 = _mm256_fmadd_pd(va, _mm256_loadu_pd(ps.add(i + 4)), _mm256_loadu_pd(pd.add(i + 4)));
        _mm256_storeu_pd(pd.add(i + 4), d1);
        let d2 = _mm256_fmadd_pd(va, _mm256_loadu_pd(ps.add(i + 8)), _mm256_loadu_pd(pd.add(i + 8)));
        _mm256_storeu_pd(pd.add(i + 8), d2);
        let d3 = _mm256_fmadd_pd(va, _mm256_loadu_pd(ps.add(i + 12)), _mm256_loadu_pd(pd.add(i + 12)));
        _mm256_storeu_pd(pd.add(i + 12), d3);
        i += 16;
    }
    while i + 4 <= n {
        let d0 = _mm256_fmadd_pd(va, _mm256_loadu_pd(ps.add(i)), _mm256_loadu_pd(pd.add(i)));
        _mm256_storeu_pd(pd.add(i), d0);
        i += 4;
    }
    while i < n {
        *pd.add(i) += a * *ps.add(i);
        i += 1;
    }
}
