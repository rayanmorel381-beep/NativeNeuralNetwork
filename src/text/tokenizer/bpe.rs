use super::errors::TokenizerError;
use crate::engine::runtime::hardware::{mmap_shared_anon, munmap};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MergePair {
    pub left: u32,
    pub right: u32,
    pub merged: u32,
}

const DEAD: i32 = -2;
const NONE: i32 = -1;
const EMPTY_KEY: u64 = u64::MAX;

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

#[inline]
fn pair_lookup(idx_key: &[u64], idx_val: &[u64], hmask: usize, a: u32, b: u32) -> Option<(u32, u32)> {
    let key = ((a as u64) << 32) | b as u64;
    let mut slot = (mix64(key) as usize) & hmask;
    loop {
        let k = idx_key[slot];
        if k == EMPTY_KEY {
            return None;
        }
        if k == key {
            let v = idx_val[slot];
            return Some(((v >> 32) as u32, (v & 0xFFFF_FFFF) as u32));
        }
        slot = (slot + 1) & hmask;
    }
}

#[inline]
fn heap_push(heap: &mut [u64], hlen: &mut usize, value: u64) -> bool {
    if *hlen >= heap.len() {
        return false;
    }
    let mut c = *hlen;
    heap[c] = value;
    *hlen += 1;
    while c > 0 {
        let p = (c - 1) / 2;
        if heap[p] <= heap[c] {
            break;
        }
        heap.swap(p, c);
        c = p;
    }
    true
}

#[inline]
fn heap_pop(heap: &mut [u64], hlen: &mut usize) -> u64 {
    let top = heap[0];
    *hlen -= 1;
    let last = *hlen;
    heap[0] = heap[last];
    let mut c = 0usize;
    loop {
        let l = 2 * c + 1;
        let r = 2 * c + 2;
        let mut s = c;
        if l < *hlen && heap[l] < heap[s] {
            s = l;
        }
        if r < *hlen && heap[r] < heap[s] {
            s = r;
        }
        if s == c {
            break;
        }
        heap.swap(c, s);
        c = s;
    }
    top
}

pub fn apply_bpe_merges(
    ids: &mut [u32],
    len: &mut usize,
    merges: &[MergePair],
    scratch: &mut [u32],
) -> Result<(), TokenizerError> {
    let n = *len;
    if scratch.len() < n {
        return Err(TokenizerError::BufferTooSmall);
    }
    if n < 2 || merges.is_empty() {
        return Ok(());
    }

    let m = merges.len();
    let h = m.saturating_mul(2).next_power_of_two().max(2);
    let hmask = h - 1;
    let heap_cap = n.saturating_mul(3).saturating_add(16);

    let prev_bytes = n.checked_mul(4).ok_or(TokenizerError::MergeCapacityExceeded)?;
    let heap_bytes = heap_cap.checked_mul(8).ok_or(TokenizerError::MergeCapacityExceeded)?;
    let key_bytes = h.checked_mul(8).ok_or(TokenizerError::MergeCapacityExceeded)?;
    let total = prev_bytes
        .checked_add(heap_bytes)
        .and_then(|x| x.checked_add(key_bytes))
        .and_then(|x| x.checked_add(key_bytes))
        .ok_or(TokenizerError::MergeCapacityExceeded)?;

    let base = mmap_shared_anon(total);
    if base.is_null() {
        return Err(TokenizerError::MergeCapacityExceeded);
    }

    let next = unsafe { core::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut i32, n) };

    let mut off = 0usize;
    let heap = unsafe { core::slice::from_raw_parts_mut(base.add(off) as *mut u64, heap_cap) };
    off += heap_bytes;
    let idx_key = unsafe { core::slice::from_raw_parts_mut(base.add(off) as *mut u64, h) };
    off += key_bytes;
    let idx_val = unsafe { core::slice::from_raw_parts_mut(base.add(off) as *mut u64, h) };
    off += key_bytes;
    let prev = unsafe { core::slice::from_raw_parts_mut(base.add(off) as *mut i32, n) };

    for slot in idx_key.iter_mut() {
        *slot = EMPTY_KEY;
    }
    for (r, mp) in merges.iter().enumerate() {
        let key = ((mp.left as u64) << 32) | mp.right as u64;
        let val = ((r as u64) << 32) | mp.merged as u64;
        let mut slot = (mix64(key) as usize) & hmask;
        loop {
            if idx_key[slot] == EMPTY_KEY {
                idx_key[slot] = key;
                idx_val[slot] = val;
                break;
            }
            if idx_key[slot] == key {
                break;
            }
            slot = (slot + 1) & hmask;
        }
    }

    for i in 0..n {
        next[i] = if i + 1 == n { NONE } else { (i + 1) as i32 };
        prev[i] = if i == 0 { NONE } else { (i - 1) as i32 };
    }

    let mut hlen = 0usize;
    let mut overflow = false;
    for i in 0..n - 1 {
        if let Some((r, _)) = pair_lookup(idx_key, idx_val, hmask, ids[i], ids[i + 1]) {
            if !heap_push(heap, &mut hlen, ((r as u64) << 32) | i as u64) {
                overflow = true;
                break;
            }
        }
    }

    while !overflow && hlen > 0 {
        let top = heap_pop(heap, &mut hlen);
        let rank = (top >> 32) as u32;
        let i = (top & 0xFFFF_FFFF) as usize;
        if next[i] == DEAD {
            continue;
        }
        let jr = next[i];
        if jr == NONE {
            continue;
        }
        let j = jr as usize;
        let (r, merged) = match pair_lookup(idx_key, idx_val, hmask, ids[i], ids[j]) {
            Some(x) => x,
            None => continue,
        };
        if r != rank {
            continue;
        }

        ids[i] = merged;
        let k = next[j];
        next[i] = k;
        if k != NONE {
            prev[k as usize] = i as i32;
        }
        next[j] = DEAD;
        prev[j] = DEAD;

        if k != NONE {
            if let Some((r2, _)) = pair_lookup(idx_key, idx_val, hmask, ids[i], ids[k as usize]) {
                if !heap_push(heap, &mut hlen, ((r2 as u64) << 32) | i as u64) {
                    overflow = true;
                    break;
                }
            }
        }
        let pi = prev[i];
        if pi != NONE {
            if let Some((r3, _)) = pair_lookup(idx_key, idx_val, hmask, ids[pi as usize], ids[i]) {
                if !heap_push(heap, &mut hlen, ((r3 as u64) << 32) | pi as u64) {
                    overflow = true;
                    break;
                }
            }
        }
    }

    if overflow {
        munmap(base, total);
        return Err(TokenizerError::MergeCapacityExceeded);
    }

    let mut out_len = 0usize;
    let mut cur: i32 = 0;
    while cur >= 0 {
        ids[out_len] = ids[cur as usize];
        out_len += 1;
        cur = next[cur as usize];
    }
    *len = out_len;

    munmap(base, total);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_naive(ids: &mut [u32], len: &mut usize, merges: &[MergePair], scratch: &mut [u32]) {
        for merge in merges {
            if *len < 2 {
                break;
            }
            let mut out_len = 0usize;
            let mut i = 0usize;
            while i < *len {
                if i + 1 < *len && ids[i] == merge.left && ids[i + 1] == merge.right {
                    scratch[out_len] = merge.merged;
                    out_len += 1;
                    i += 2;
                } else {
                    scratch[out_len] = ids[i];
                    out_len += 1;
                    i += 1;
                }
            }
            ids[..out_len].copy_from_slice(&scratch[..out_len]);
            *len = out_len;
        }
    }

    fn check(seq: &[u32], merges: &[MergePair]) {
        let mut a = [0u32; 128];
        let mut b = [0u32; 128];
        a[..seq.len()].copy_from_slice(seq);
        b[..seq.len()].copy_from_slice(seq);
        let mut la = seq.len();
        let mut lb = seq.len();
        let mut sa = [0u32; 128];
        let mut sb = [0u32; 128];
        apply_naive(&mut a, &mut la, merges, &mut sa);
        apply_bpe_merges(&mut b, &mut lb, merges, &mut sb).unwrap();
        assert_eq!(la, lb);
        assert_eq!(a[..la], b[..lb]);
    }

    #[test]
    fn overlap_same_pair() {
        let merges = [MergePair { left: 7, right: 7, merged: 256 }];
        let buf = [7u32; 11];
        for count in 1..12usize {
            check(&buf[..count], &merges);
        }
    }

    #[test]
    fn chained_merges() {
        let merges = [
            MergePair { left: 97, right: 98, merged: 256 },
            MergePair { left: 256, right: 256, merged: 257 },
            MergePair { left: 98, right: 97, merged: 258 },
        ];
        let seq = [97u32, 98, 97, 98, 97, 98];
        check(&seq, &merges);
    }

    #[test]
    fn random_match_naive() {
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut rng = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut merges = [MergePair { left: 0, right: 0, merged: 0 }; 10];
        let mut seqbuf = [0u32; 40];
        for _ in 0..400 {
            let base = 6u32;
            let n_merges = (rng() % 10) as usize + 1;
            for r in 0..n_merges {
                let max_id = base + r as u32;
                merges[r] = MergePair {
                    left: (rng() as u32) % max_id,
                    right: (rng() as u32) % max_id,
                    merged: base + r as u32,
                };
            }
            let seq_len = (rng() % 38) as usize + 2;
            for s in seqbuf.iter_mut().take(seq_len) {
                *s = (rng() as u32) % base;
            }
            check(&seqbuf[..seq_len], &merges[..n_merges]);
        }
    }
}
