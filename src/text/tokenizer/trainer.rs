use super::bpe::MergePair;
use super::errors::TokenizerError;
use crate::engine::runtime::hardware::{mmap_shared_anon, munmap};

pub struct TrainerScratch<'a> {
    pub symbols: &'a mut [u32],
    pub next: &'a mut [i32],
    pub prev: &'a mut [i32],
}

#[inline]
fn hash_pair(a: u32, b: u32) -> u64 {
    let mut h = 1469598103934665603u64;
    h = (h ^ a as u64).wrapping_mul(1099511628211);
    h = (h ^ b as u64).wrapping_mul(1099511628211);
    h
}

pub struct TrainerConfig {
    pub initial_vocab_size: u32,
    pub target_merges: usize,
    pub min_pair_count: u32,
}

const DEAD: i32 = i32::MIN;

#[inline]
fn next_pow2(x: usize) -> usize {
    let mut p = 1usize;
    while p < x {
        p <<= 1;
    }
    p
}

struct Pairs<'a> {
    pl: &'a mut [u32],
    pr: &'a mut [u32],
    pc: &'a mut [i32],
    ph: &'a mut [i32],
    bnext: &'a mut [i32],
    bprev: &'a mut [i32],
    hash: &'a mut [i32],
    posn: &'a mut [i32],
    posp: &'a mut [i32],
    bhead: &'a mut [i32],
    mask: usize,
    nb_pairs: usize,
    cap_pairs: usize,
    max_count: usize,
}

impl Pairs<'_> {
    fn get_or_create(&mut self, a: u32, b: u32) -> Option<usize> {
        let mut slot = (hash_pair(a, b) as usize) & self.mask;
        loop {
            let e = self.hash[slot];
            if e < 0 {
                if self.nb_pairs >= self.cap_pairs {
                    return None;
                }
                let pid = self.nb_pairs;
                self.nb_pairs += 1;
                self.pl[pid] = a;
                self.pr[pid] = b;
                self.pc[pid] = 0;
                self.ph[pid] = -1;
                self.bnext[pid] = -1;
                self.bprev[pid] = -1;
                self.hash[slot] = pid as i32;
                return Some(pid);
            }
            let k = e as usize;
            if self.pl[k] == a && self.pr[k] == b {
                return Some(k);
            }
            slot = (slot + 1) & self.mask;
        }
    }

    fn lookup(&self, a: u32, b: u32) -> Option<usize> {
        let mut slot = (hash_pair(a, b) as usize) & self.mask;
        loop {
            let e = self.hash[slot];
            if e < 0 {
                return None;
            }
            let k = e as usize;
            if self.pl[k] == a && self.pr[k] == b {
                return Some(k);
            }
            slot = (slot + 1) & self.mask;
        }
    }

    fn pos_add(&mut self, pid: usize, pos: usize) {
        let head = self.ph[pid];
        self.posn[pos] = head;
        self.posp[pos] = -1;
        if head >= 0 {
            self.posp[head as usize] = pos as i32;
        }
        self.ph[pid] = pos as i32;
    }

    fn pos_remove(&mut self, pid: usize, pos: usize) {
        let pn = self.posn[pos];
        let pp = self.posp[pos];
        if pp >= 0 {
            self.posn[pp as usize] = pn;
        } else {
            self.ph[pid] = pn;
        }
        if pn >= 0 {
            self.posp[pn as usize] = pp;
        }
    }

    fn bucket_add(&mut self, pid: usize, c: usize) {
        let head = self.bhead[c];
        self.bnext[pid] = head;
        self.bprev[pid] = -1;
        if head >= 0 {
            self.bprev[head as usize] = pid as i32;
        }
        self.bhead[c] = pid as i32;
    }

    fn bucket_remove(&mut self, pid: usize, c: usize) {
        let bn = self.bnext[pid];
        let bp = self.bprev[pid];
        if bp >= 0 {
            self.bnext[bp as usize] = bn;
        } else {
            self.bhead[c] = bn;
        }
        if bn >= 0 {
            self.bprev[bn as usize] = bp;
        }
    }

    fn inc_pair(&mut self, pid: usize, pos: usize) {
        self.pos_add(pid, pos);
        let c = self.pc[pid];
        if c > 0 {
            self.bucket_remove(pid, c as usize);
        }
        let nc = c + 1;
        self.pc[pid] = nc;
        self.bucket_add(pid, nc as usize);
        if nc as usize > self.max_count {
            self.max_count = nc as usize;
        }
    }

    fn dec_pair(&mut self, pid: usize, pos: usize) {
        self.pos_remove(pid, pos);
        let c = self.pc[pid];
        self.bucket_remove(pid, c as usize);
        let nc = c - 1;
        self.pc[pid] = nc;
        if nc > 0 {
            self.bucket_add(pid, nc as usize);
        }
    }
}

pub fn train_bpe(
    corpus_len: usize,
    cfg: &TrainerConfig,
    scratch: TrainerScratch<'_>,
    out_merges: &mut [MergePair],
) -> Result<usize, TokenizerError> {
    let n = corpus_len;
    if n < 2 {
        return Err(TokenizerError::TrainerNotEnoughBytes);
    }
    let TrainerScratch { symbols, next, prev } = scratch;
    if symbols.len() < n || next.len() < n || prev.len() < n {
        return Err(TokenizerError::TrainerCapacityExceeded);
    }
    if out_merges.len() < cfg.target_merges {
        return Err(TokenizerError::TrainerCapacityExceeded);
    }

    let cap_pairs = n.saturating_mul(3).saturating_add(512);
    let hash_cap = next_pow2(cap_pairs.saturating_mul(2).max(1024));
    let bucket_cap = n + 2;

    let o_pl = 0usize;
    let o_pr = o_pl + cap_pairs;
    let o_pc = o_pr + cap_pairs;
    let o_ph = o_pc + cap_pairs;
    let o_bn = o_ph + cap_pairs;
    let o_bp = o_bn + cap_pairs;
    let o_hash = o_bp + cap_pairs;
    let o_posn = o_hash + hash_cap;
    let o_posp = o_posn + n;
    let o_bhead = o_posp + n;
    let o_temp = o_bhead + bucket_cap;
    let total_elems = o_temp + n;
    let total_bytes = total_elems * 4;

    let base = mmap_shared_anon(total_bytes);
    if base.is_null() {
        return Err(TokenizerError::TrainerCapacityExceeded);
    }

    let pl = unsafe { core::slice::from_raw_parts_mut(base.add(o_pl * 4) as *mut u32, cap_pairs) };
    let pr = unsafe { core::slice::from_raw_parts_mut(base.add(o_pr * 4) as *mut u32, cap_pairs) };
    let pc = unsafe { core::slice::from_raw_parts_mut(base.add(o_pc * 4) as *mut i32, cap_pairs) };
    let ph = unsafe { core::slice::from_raw_parts_mut(base.add(o_ph * 4) as *mut i32, cap_pairs) };
    let bnext = unsafe { core::slice::from_raw_parts_mut(base.add(o_bn * 4) as *mut i32, cap_pairs) };
    let bprev = unsafe { core::slice::from_raw_parts_mut(base.add(o_bp * 4) as *mut i32, cap_pairs) };
    let hash = unsafe { core::slice::from_raw_parts_mut(base.add(o_hash * 4) as *mut i32, hash_cap) };
    let posn = unsafe { core::slice::from_raw_parts_mut(base.add(o_posn * 4) as *mut i32, n) };
    let posp = unsafe { core::slice::from_raw_parts_mut(base.add(o_posp * 4) as *mut i32, n) };
    let bhead = unsafe { core::slice::from_raw_parts_mut(base.add(o_bhead * 4) as *mut i32, bucket_cap) };
    let temp = unsafe { core::slice::from_raw_parts_mut(base.add(o_temp * 4) as *mut i32, n) };

    for slot in hash.iter_mut() {
        *slot = -1;
    }
    for slot in bhead.iter_mut() {
        *slot = -1;
    }
    for i in 0..n {
        prev[i] = if i == 0 { -1 } else { i as i32 - 1 };
        next[i] = if i + 1 == n { -1 } else { i as i32 + 1 };
    }

    let mut pairs = Pairs {
        pl,
        pr,
        pc,
        ph,
        bnext,
        bprev,
        hash,
        posn,
        posp,
        bhead,
        mask: hash_cap - 1,
        nb_pairs: 0,
        cap_pairs,
        max_count: 0,
    };

    let mut pos = 0usize;
    while pos + 1 < n {
        let a = symbols[pos];
        let b = symbols[pos + 1];
        match pairs.get_or_create(a, b) {
            Some(pid) => pairs.inc_pair(pid, pos),
            None => break,
        }
        pos += 1;
    }

    let mut produced = 0usize;
    let mut next_id = cfg.initial_vocab_size;
    let min_c = cfg.min_pair_count as usize;

    while produced < cfg.target_merges {
        while pairs.max_count >= 1 && pairs.bhead[pairs.max_count] < 0 {
            pairs.max_count -= 1;
        }
        if pairs.max_count == 0 || pairs.max_count < min_c {
            break;
        }
        let pid = pairs.bhead[pairs.max_count] as usize;
        let win_count = pairs.pc[pid] as usize;
        pairs.bucket_remove(pid, win_count);
        let left = pairs.pl[pid];
        let right = pairs.pr[pid];
        let merged = next_id;
        next_id += 1;
        out_merges[produced] = MergePair { left, right, merged };
        produced += 1;

        let mut cnt = 0usize;
        let mut cur = pairs.ph[pid];
        while cur >= 0 && cnt < n {
            temp[cnt] = cur;
            cnt += 1;
            cur = pairs.posn[cur as usize];
        }
        temp[..cnt].sort_unstable();

        for slot in temp.iter().take(cnt) {
            let i = *slot as usize;
            if prev[i] == DEAD {
                continue;
            }
            let jj = next[i];
            if jj < 0 {
                continue;
            }
            let j = jj as usize;
            if symbols[i] != left || symbols[j] != right {
                continue;
            }
            let p = prev[i];
            let k = next[j];
            if p >= 0 {
                let pp = p as usize;
                if let Some(lpid) = pairs.lookup(symbols[pp], symbols[i]) {
                    if lpid != pid {
                        pairs.dec_pair(lpid, pp);
                    }
                }
            }
            if k >= 0 {
                let kk = k as usize;
                if let Some(rpid) = pairs.lookup(symbols[j], symbols[kk]) {
                    if rpid != pid {
                        pairs.dec_pair(rpid, j);
                    }
                }
            }
            symbols[i] = merged;
            next[i] = k;
            if k >= 0 {
                prev[k as usize] = i as i32;
            }
            prev[j] = DEAD;
            if p >= 0 {
                let pp = p as usize;
                if let Some(np) = pairs.get_or_create(symbols[pp], merged) {
                    pairs.inc_pair(np, pp);
                }
            }
            if k >= 0 {
                let kk = k as usize;
                if let Some(np) = pairs.get_or_create(merged, symbols[kk]) {
                    pairs.inc_pair(np, i);
                }
            }
        }

        pairs.pc[pid] = 0;
        pairs.ph[pid] = -1;
    }

    munmap(base, total_bytes);
    Ok(produced)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_pair(seq: &[u32], l: u32, r: u32) -> usize {
        let mut c = 0usize;
        let mut i = 0usize;
        while i + 1 < seq.len() {
            if seq[i] == l && seq[i + 1] == r {
                c += 1;
            }
            i += 1;
        }
        c
    }

    fn naive_best(seq: &[u32]) -> (u32, u32, usize) {
        let mut bl = 0u32;
        let mut br = 0u32;
        let mut bc = 0usize;
        let mut i = 0usize;
        while i + 1 < seq.len() {
            let a = seq[i];
            let b = seq[i + 1];
            let c = count_pair(seq, a, b);
            if c > bc {
                bc = c;
                bl = a;
                br = b;
            }
            i += 1;
        }
        (bl, br, bc)
    }

    fn apply(seq: &mut [u32], len: usize, l: u32, r: u32, m: u32) -> usize {
        let mut w = 0usize;
        let mut i = 0usize;
        while i < len {
            if i + 1 < len && seq[i] == l && seq[i + 1] == r {
                seq[w] = m;
                w += 1;
                i += 2;
            } else {
                seq[w] = seq[i];
                w += 1;
                i += 1;
            }
        }
        w
    }

    fn run_case(corpus: &[u8], target_merges: usize, min_pair_count: u32) {
        let n = corpus.len();
        let mut symbols = [0u32; 256];
        let mut next = [0i32; 256];
        let mut prev = [0i32; 256];
        let mut merges = [MergePair { left: 0, right: 0, merged: 0 }; 200];
        for (i, b) in corpus.iter().enumerate() {
            symbols[i] = *b as u32;
        }
        let cfg = TrainerConfig {
            initial_vocab_size: 256,
            target_merges,
            min_pair_count,
        };
        let produced = train_bpe(
            n,
            &cfg,
            TrainerScratch {
                symbols: &mut symbols[..n],
                next: &mut next[..n],
                prev: &mut prev[..n],
            },
            &mut merges[..target_merges],
        )
        .unwrap();

        let mut seq = [0u32; 256];
        for (i, b) in corpus.iter().enumerate() {
            seq[i] = *b as u32;
        }
        let mut len = n;
        for idx in 0..produced {
            let mp = merges[idx];
            let (_, _, best_c) = naive_best(&seq[..len]);
            let lr_c = count_pair(&seq[..len], mp.left, mp.right);
            assert_eq!(lr_c, best_c, "merge {idx}: chosen pair count {lr_c} != global max {best_c}");
            assert!(best_c >= min_pair_count as usize, "merge {idx}: count {best_c} below min");
            assert_eq!(mp.merged, 256 + idx as u32, "merge {idx}: merged id not sequential");
            len = apply(&mut seq, len, mp.left, mp.right, mp.merged);
            assert_eq!(count_pair(&seq[..len], mp.left, mp.right), 0, "merge {idx}: residual pair");
        }
        let (_, _, final_best) = naive_best(&seq[..len]);
        assert!(
            produced == target_merges || final_best < min_pair_count as usize,
            "stopped early with a valid merge still available: best={final_best}"
        );
    }

    #[test]
    fn train_bpe_matches_naive_greedy() {
        run_case(b"aaaaabbbbb abababab abcabcabc aabbccddee the theother xyzxyzxy", 64, 2);
    }

    #[test]
    fn train_bpe_handles_long_runs() {
        run_case(b"aaaaaaaaaaaaaaaaaaaa bbbbbbbbbb cccccc", 32, 2);
    }

    #[test]
    fn train_bpe_respects_min_count() {
        run_case(b"the quick brown fox the the the", 40, 3);
    }

    #[test]
    fn train_bpe_target_cap() {
        run_case(b"abababababababababababababababab", 3, 1);
    }
}
