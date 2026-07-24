use super::bpe::{apply_bpe_merges, MergePair};
use super::errors::TokenizerError;
use super::vocab::{encode_bytes_checked, Vocab};
use crate::engine::runtime::{thread_parallel_for, SyncMutPtr};
use core::sync::atomic::{AtomicBool, Ordering};

const SEG_TARGET: usize = 256 * 1024;
const MAX_SEG: usize = 256;

pub fn encode_with_bpe(
    text: &[u8],
    vocab: &Vocab,
    merges: &[MergePair],
    out: &mut [u32],
    scratch: &mut [u32],
) -> Result<usize, TokenizerError> {
    let n_seg = (text.len() / SEG_TARGET).min(MAX_SEG);
    if n_seg < 2 || merges.is_empty() {
        return encode_with_bpe_seq(text, vocab, merges, out, scratch);
    }
    encode_with_bpe_segmented(text, vocab, merges, out, scratch, n_seg)
}

fn encode_with_bpe_seq(
    text: &[u8],
    vocab: &Vocab,
    merges: &[MergePair],
    out: &mut [u32],
    scratch: &mut [u32],
) -> Result<usize, TokenizerError> {
    let n = encode_bytes_checked(text, vocab, out)?;
    let mut len = n;
    apply_bpe_merges(out, &mut len, merges, scratch)?;
    Ok(len)
}

fn encode_with_bpe_segmented(
    text: &[u8],
    vocab: &Vocab,
    merges: &[MergePair],
    out: &mut [u32],
    scratch: &mut [u32],
    n_seg: usize,
) -> Result<usize, TokenizerError> {
    let mut base = [0u32; 256];
    for (b, slot) in base.iter_mut().enumerate() {
        *slot = vocab.encode_byte(b as u8);
    }
    let mut barrier = [true; 256];
    for m in merges {
        for b in 0..256 {
            if barrier[b] && (base[b] == m.left || base[b] == m.right) {
                barrier[b] = false;
            }
        }
    }

    let len_t = text.len();
    let mut cuts = [0usize; MAX_SEG + 1];
    let mut nc = 1usize;
    for s in 1..n_seg {
        let target = s * len_t / n_seg;
        let mut pos = target;
        while pos < len_t && !barrier[text[pos] as usize] {
            pos += 1;
        }
        if pos < len_t && pos > cuts[nc - 1] {
            cuts[nc] = pos;
            nc += 1;
        }
    }
    cuts[nc] = len_t;
    if nc < 2 {
        return encode_with_bpe_seq(text, vocab, merges, out, scratch);
    }

    let mut seg_len = [0usize; MAX_SEG];
    let out_ptr = SyncMutPtr(out.as_mut_ptr());
    let scr_ptr = SyncMutPtr(scratch.as_mut_ptr());
    let slen_ptr = SyncMutPtr(seg_len.as_mut_ptr());
    let failed = AtomicBool::new(false);
    let failed_ref = &failed;
    thread_parallel_for(nc, &move |s| {
        let _ = (&out_ptr, &scr_ptr, &slen_ptr);
        let a = cuts[s];
        let b = cuts[s + 1];
        let seg_text = &text[a..b];
        let seg_out = unsafe { core::slice::from_raw_parts_mut(out_ptr.0.add(a), b - a) };
        let seg_scr = unsafe { core::slice::from_raw_parts_mut(scr_ptr.0.add(a), b - a) };
        match encode_bytes_checked(seg_text, vocab, seg_out) {
            Ok(nn) => {
                let mut l = nn;
                match apply_bpe_merges(seg_out, &mut l, merges, seg_scr) {
                    Ok(()) => unsafe {
                        *slen_ptr.0.add(s) = l;
                    },
                    Err(_) => failed_ref.store(true, Ordering::Relaxed),
                }
            }
            Err(_) => failed_ref.store(true, Ordering::Relaxed),
        }
    });

    if failed.load(Ordering::Relaxed) {
        return encode_with_bpe_seq(text, vocab, merges, out, scratch);
    }

    let mut total = 0usize;
    for s in 0..nc {
        let a = cuts[s];
        let l = seg_len[s];
        if a != total {
            out.copy_within(a..a + l, total);
        }
        total += l;
    }
    Ok(total)
}

pub fn decode_with_vocab(
    ids: &[u32],
    vocab: &Vocab,
    out: &mut [u8],
) -> Result<usize, TokenizerError> {
    super::vocab::decode_ids_utf8(ids, vocab, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segmented_matches_sequential() {
        let entries: [(&[u8], u32); 4] =
            [(&b"a"[..], 0), (&b"b"[..], 1), (&b"c"[..], 2), (&b"x"[..], 3)];
        let vocab = Vocab::new(&entries, 999).unwrap();
        let merges = [
            MergePair { left: 0, right: 1, merged: 4 },
            MergePair { left: 4, right: 2, merged: 5 },
        ];
        let inputs: [&[u8]; 4] = [
            b"abcxabcxabc",
            b"xxabcabcxxabcx",
            b"abcabcabcxabc",
            b"xabcabcabcabcabcx",
        ];
        for input in inputs {
            let mut out_seq = [0u32; 128];
            let mut scr_seq = [0u32; 128];
            let n_seq =
                encode_with_bpe_seq(input, &vocab, &merges, &mut out_seq, &mut scr_seq).unwrap();
            for nseg in 2..=6 {
                let mut out_p = [0u32; 128];
                let mut scr_p = [0u32; 128];
                let n_p =
                    encode_with_bpe_segmented(input, &vocab, &merges, &mut out_p, &mut scr_p, nseg)
                        .unwrap();
                assert_eq!(n_p, n_seq, "longueur differente nseg={}", nseg);
                assert_eq!(out_p[..n_p], out_seq[..n_seq], "tokens differents nseg={}", nseg);
            }
        }
    }

    #[test]
    fn segmented_without_barrier_matches_sequential() {
        let entries: [(&[u8], u32); 2] = [(&b"a"[..], 0), (&b"b"[..], 1)];
        let vocab = Vocab::new(&entries, 999).unwrap();
        let merges = [MergePair { left: 0, right: 1, merged: 4 }];
        let input: &[u8] = b"abababab";
        let mut out_seq = [0u32; 64];
        let mut scr_seq = [0u32; 64];
        let n_seq =
            encode_with_bpe_seq(input, &vocab, &merges, &mut out_seq, &mut scr_seq).unwrap();
        let mut out_p = [0u32; 64];
        let mut scr_p = [0u32; 64];
        let n_p =
            encode_with_bpe_segmented(input, &vocab, &merges, &mut out_p, &mut scr_p, 4).unwrap();
        assert_eq!(n_p, n_seq);
        assert_eq!(out_p[..n_p], out_seq[..n_seq]);
    }

    #[test]
    fn segmented_matches_sequential_random() {
        let entries: [(&[u8], u32); 8] = [
            (&b"\n"[..], 7),
            (&b" "[..], 6),
            (&b"a"[..], 0),
            (&b"b"[..], 1),
            (&b"c"[..], 2),
            (&b"d"[..], 3),
            (&b"e"[..], 4),
            (&b"f"[..], 5),
        ];
        let vocab = Vocab::new(&entries, 999).unwrap();
        let merges = [
            MergePair { left: 0, right: 1, merged: 8 },
            MergePair { left: 2, right: 3, merged: 9 },
            MergePair { left: 8, right: 9, merged: 10 },
            MergePair { left: 4, right: 5, merged: 11 },
            MergePair { left: 0, right: 0, merged: 12 },
            MergePair { left: 10, right: 11, merged: 13 },
            MergePair { left: 12, right: 2, merged: 14 },
            MergePair { left: 1, right: 1, merged: 15 },
        ];
        let symbols: [u8; 8] = [b'a', b'b', b'c', b'd', b'e', b'f', b' ', b'\n'];
        let mut input = [0u8; 4000];
        let mut state: u64 = 0x1234_5678_9abc_def0;
        for slot in input.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = ((state >> 33) as usize) % 10;
            let idx = if pick < 6 { pick } else { 6 + (pick & 1) };
            *slot = symbols[idx];
        }
        let mut out_seq = [0u32; 8192];
        let mut scr_seq = [0u32; 8192];
        let n_seq =
            encode_with_bpe_seq(&input, &vocab, &merges, &mut out_seq, &mut scr_seq).unwrap();
        for nseg in 2..=9 {
            let mut out_p = [0u32; 8192];
            let mut scr_p = [0u32; 8192];
            let n_p =
                encode_with_bpe_segmented(&input, &vocab, &merges, &mut out_p, &mut scr_p, nseg)
                    .unwrap();
            assert_eq!(n_p, n_seq, "longueur differente nseg={}", nseg);
            assert_eq!(out_p[..n_p], out_seq[..n_seq], "tokens differents nseg={}", nseg);
        }
    }
}
