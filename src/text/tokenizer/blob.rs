use super::errors::TokenizerError;
use super::bpe::MergePair;

pub const VOCAB_MAGIC: [u8; 4] = *b"VOCB";
pub const VOCAB_VERSION: u16 = 1;
pub const MERGES_MAGIC: [u8; 4] = *b"MRGS";
pub const MERGES_VERSION: u16 = 1;

fn read_u16(blob: &[u8], at: &mut usize) -> Result<u16, TokenizerError> {
    if *at + 2 > blob.len() {
        return Err(TokenizerError::BlobTruncated);
    }
    let v = u16::from_le_bytes([blob[*at], blob[*at + 1]]);
    *at += 2;
    Ok(v)
}

fn read_u32(blob: &[u8], at: &mut usize) -> Result<u32, TokenizerError> {
    if *at + 4 > blob.len() {
        return Err(TokenizerError::BlobTruncated);
    }
    let v = u32::from_le_bytes([blob[*at], blob[*at + 1], blob[*at + 2], blob[*at + 3]]);
    *at += 4;
    Ok(v)
}

fn read_slice<'a>(blob: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8], TokenizerError> {
    if *at + len > blob.len() {
        return Err(TokenizerError::BlobTruncated);
    }
    let s = &blob[*at..*at + len];
    *at += len;
    Ok(s)
}

pub fn parse_vocab_blob<'a>(
    blob: &'a [u8],
    entries: &mut [(&'a [u8], u32)],
) -> Result<(usize, u32), TokenizerError> {
    let mut at = 0usize;
    let magic = read_slice(blob, &mut at, 4)?;
    if magic != VOCAB_MAGIC {
        return Err(TokenizerError::BlobBadMagic);
    }
    let version = read_u16(blob, &mut at)?;
    if version != VOCAB_VERSION {
        return Err(TokenizerError::BlobBadVersion);
    }
    let unk_id = read_u32(blob, &mut at)?;
    let nb = read_u32(blob, &mut at)? as usize;
    if entries.len() < nb {
        return Err(TokenizerError::BlobEntriesBufferTooSmall);
    }
    for slot in entries.iter_mut().take(nb) {
        let len = read_u16(blob, &mut at)? as usize;
        let bytes = read_slice(blob, &mut at, len)?;
        let id = read_u32(blob, &mut at)?;
        *slot = (bytes, id);
    }
    if nb > 1 {
        entries[..nb].sort_unstable_by(|a, b| a.0.cmp(b.0));
    }
    Ok((nb, unk_id))
}

pub fn parse_merges_blob(
    blob: &[u8],
    merges: &mut [MergePair],
) -> Result<usize, TokenizerError> {
    let mut at = 0usize;
    let magic = read_slice(blob, &mut at, 4)?;
    if magic != MERGES_MAGIC {
        return Err(TokenizerError::BlobBadMagic);
    }
    let version = read_u16(blob, &mut at)?;
    if version != MERGES_VERSION {
        return Err(TokenizerError::BlobBadVersion);
    }
    let nb = read_u32(blob, &mut at)? as usize;
    if merges.len() < nb {
        return Err(TokenizerError::BlobEntriesBufferTooSmall);
    }
    for slot in merges.iter_mut().take(nb) {
        let left = read_u32(blob, &mut at)?;
        let right = read_u32(blob, &mut at)?;
        let merged = read_u32(blob, &mut at)?;
        *slot = MergePair { left, right, merged };
    }
    Ok(nb)
}

fn write_u16(out: &mut [u8], at: &mut usize, v: u16) -> Result<(), TokenizerError> {
    if *at + 2 > out.len() {
        return Err(TokenizerError::BufferTooSmall);
    }
    let b = v.to_le_bytes();
    out[*at] = b[0];
    out[*at + 1] = b[1];
    *at += 2;
    Ok(())
}

fn write_u32(out: &mut [u8], at: &mut usize, v: u32) -> Result<(), TokenizerError> {
    if *at + 4 > out.len() {
        return Err(TokenizerError::BufferTooSmall);
    }
    let b = v.to_le_bytes();
    out[*at] = b[0];
    out[*at + 1] = b[1];
    out[*at + 2] = b[2];
    out[*at + 3] = b[3];
    *at += 4;
    Ok(())
}

fn write_bytes(out: &mut [u8], at: &mut usize, src: &[u8]) -> Result<(), TokenizerError> {
    if *at + src.len() > out.len() {
        return Err(TokenizerError::BufferTooSmall);
    }
    out[*at..*at + src.len()].copy_from_slice(src);
    *at += src.len();
    Ok(())
}

pub fn write_vocab_blob(
    entries: &[(&[u8], u32)],
    unk_id: u32,
    out: &mut [u8],
) -> Result<usize, TokenizerError> {
    let mut at = 0usize;
    write_bytes(out, &mut at, &VOCAB_MAGIC)?;
    write_u16(out, &mut at, VOCAB_VERSION)?;
    write_u32(out, &mut at, unk_id)?;
    write_u32(out, &mut at, entries.len() as u32)?;
    for (bytes, id) in entries {
        if bytes.len() > u16::MAX as usize {
            return Err(TokenizerError::BufferTooSmall);
        }
        write_u16(out, &mut at, bytes.len() as u16)?;
        write_bytes(out, &mut at, bytes)?;
        write_u32(out, &mut at, *id)?;
    }
    Ok(at)
}

pub fn write_merges_blob(
    merges: &[MergePair],
    out: &mut [u8],
) -> Result<usize, TokenizerError> {
    let mut at = 0usize;
    write_bytes(out, &mut at, &MERGES_MAGIC)?;
    write_u16(out, &mut at, MERGES_VERSION)?;
    write_u32(out, &mut at, merges.len() as u32)?;
    for m in merges {
        write_u32(out, &mut at, m.left)?;
        write_u32(out, &mut at, m.right)?;
        write_u32(out, &mut at, m.merged)?;
    }
    Ok(at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::vocab::Vocab;

    #[test]
    fn parse_vocab_blob_sorts_entries_for_binary_search() {
        let entries = [
            (b"b" as &[u8], 1u32),
            (b"a" as &[u8], 2u32),
            (b"c" as &[u8], 3u32),
        ];
        let mut raw = [0u8; 256];
        let mut at = 0usize;
        write_bytes(&mut raw, &mut at, &VOCAB_MAGIC).unwrap();
        write_u16(&mut raw, &mut at, VOCAB_VERSION).unwrap();
        write_u32(&mut raw, &mut at, 0).unwrap();
        write_u32(&mut raw, &mut at, entries.len() as u32).unwrap();
        for (token, id) in entries.iter() {
            write_u16(&mut raw, &mut at, token.len() as u16).unwrap();
            write_bytes(&mut raw, &mut at, token).unwrap();
            write_u32(&mut raw, &mut at, *id).unwrap();
        }
        let mut parsed_entries = [(&[][..], 0u32); 8];
        let (nb, unk_id) = parse_vocab_blob(&raw[..at], &mut parsed_entries).unwrap();
        assert_eq!(nb, entries.len());
        assert_eq!(unk_id, 0);

        let vocab = Vocab::new(&parsed_entries[..nb], unk_id).unwrap();
        assert_eq!(vocab.token_to_id(b"a"), 2);
        assert_eq!(vocab.token_to_id(b"b"), 1);
        assert_eq!(vocab.token_to_id(b"c"), 3);
    }

    #[test]
    fn decode_ids_bpe_resolves_merges_by_id_in_any_order() {
        let merges = [
            MergePair { left: 98, right: 97, merged: 258 },
            MergePair { left: 256, right: 256, merged: 257 },
            MergePair { left: 97, right: 98, merged: 256 },
        ];
        let ids = [256u32, 258u32];
        let mut out = [0u8; 64];
        let written = super::super::vocab::decode_ids_bpe(&ids, &merges, &mut out).unwrap();
        assert_eq!(&out[..written], b"abba");
    }
}

