use super::errors::TokenizerError;

pub struct Vocab<'a> {
    entries: &'a [(&'a [u8], u32)],
    unk_id: u32,
}

impl<'a> Vocab<'a> {
    pub fn new(entries: &'a [(&'a [u8], u32)], unk_id: u32) -> Result<Self, TokenizerError> {
        if entries.is_empty() {
            return Err(TokenizerError::VocabEmpty);
        }
        Ok(Self { entries, unk_id })
    }

    pub fn encode_byte(&self, byte: u8) -> u32 {
        self.token_to_id(&[byte])
    }

    pub fn encode_byte_checked(&self, byte: u8) -> Result<u32, TokenizerError> {
        let id = self.encode_byte(byte);
        if id == self.unk_id() {
            Err(TokenizerError::UnknownByte)
        } else {
            Ok(id)
        }
    }

    pub fn token_to_id(&self, token: &[u8]) -> u32 {
        let mut lo = 0usize;
        let mut hi = self.entries.len();
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.entries[mid].0.cmp(token) {
                core::cmp::Ordering::Equal => return self.entries[mid].1,
                core::cmp::Ordering::Less => lo = mid + 1,
                core::cmp::Ordering::Greater => hi = mid,
            }
        }
        self.unk_id
    }

    pub fn id_to_token(&self, id: u32) -> Option<&[u8]> {
        for &(k, v) in self.entries {
            if v == id {
                return Some(k);
            }
        }
        None
    }

    pub fn vocab_size(&self) -> usize {
        self.entries.len()
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }
}

pub fn encode_bytes(
    input: &[u8],
    vocab: &Vocab,
    out: &mut [u32],
) -> Result<usize, TokenizerError> {
    if vocab.vocab_size() == 0 {
        return Err(TokenizerError::VocabEmpty);
    }
    if out.len() < input.len() {
        return Err(TokenizerError::BufferTooSmall);
    }
    for (i, &b) in input.iter().enumerate() {
        out[i] = vocab.encode_byte_checked(b).unwrap_or_else(|_| vocab.unk_id());
    }
    Ok(input.len())
}

pub fn encode_bytes_checked(
    input: &[u8],
    vocab: &Vocab,
    out: &mut [u32],
) -> Result<usize, TokenizerError> {
    if vocab.vocab_size() == 0 {
        return Err(TokenizerError::VocabEmpty);
    }
    if out.len() < input.len() {
        return Err(TokenizerError::BufferTooSmall);
    }
    let count = encode_bytes(input, vocab, out)?;
    for &id in out.iter().take(input.len()) {
        if id == vocab.unk_id() {
            return Err(TokenizerError::UnknownByte);
        }
    }
    Ok(count)
}

pub fn decode_ids(
    ids: &[u32],
    vocab: &Vocab,
    out: &mut [u8],
) -> Result<usize, TokenizerError> {
    let mut written = 0usize;
    for &id in ids {
        let tok = vocab.id_to_token(id).unwrap_or(b"?");
        if written + tok.len() > out.len() {
            return Err(TokenizerError::BufferTooSmall);
        }
        out[written..written + tok.len()].copy_from_slice(tok);
        written += tok.len();
    }
    Ok(written)
}

pub fn decode_ids_utf8(
    ids: &[u32],
    vocab: &Vocab,
    out: &mut [u8],
) -> Result<usize, TokenizerError> {
    let written = decode_ids(ids, vocab, out)?;
    if core::str::from_utf8(&out[..written]).is_ok() {
        Ok(written)
    } else {
        Err(TokenizerError::InvalidUtf8)
    }
}

fn resolve_merge<'a>(merges: &'a [super::bpe::MergePair], merged_id: u32) -> Option<&'a super::bpe::MergePair> {
    merges.iter().find(|m| m.merged == merged_id)
}

pub fn decode_ids_bpe(
    ids: &[u32],
    merges: &[super::bpe::MergePair],
    out: &mut [u8],
) -> Result<usize, TokenizerError> {
    let mut stack = [0u32; 4096];
    let mut written = 0usize;
    for &id in ids {
        let mut sp = 0usize;
        stack[sp] = id;
        sp += 1;
        while sp > 0 {
            sp -= 1;
            let t = stack[sp];
            if (t as usize) < 256 {
                if written >= out.len() {
                    return Err(TokenizerError::BufferTooSmall);
                }
                out[written] = t as u8;
                written += 1;
            } else {
                let m = match resolve_merge(merges, t) {
                    Some(m) => *m,
                    None => {
                        if written >= out.len() {
                            return Err(TokenizerError::BufferTooSmall);
                        }
                        out[written] = b'?';
                        written += 1;
                        continue;
                    }
                };
                if sp + 2 > stack.len() {
                    return Err(TokenizerError::BufferTooSmall);
                }
                stack[sp] = m.right;
                sp += 1;
                stack[sp] = m.left;
                sp += 1;
            }
        }
    }
    Ok(written)
}
