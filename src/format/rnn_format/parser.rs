use crate::security::crypto::constant_time_eq;
use crate::format::model_format::{
    BLOB_BIASES, BLOB_CONV_SPEC, BLOB_LAYER_META, BLOB_NEURON_POSITIONS, BLOB_RUNTIME_INPUT,
    BLOB_TRAINING_LOG, BLOB_WEIGHTS, LMLP_CONFIG_BLOB_DATA, LMLP_GRAPH_BLOB_DATA,
    LMLP_TENSORS_BLOB_DATA, LMLP_WEIGHTS_BLOB_DATA,
    TOKENIZER_MERGES_BLOB_DATA, TOKENIZER_VOCAB_BLOB_DATA, TRAIN_CONFIG_BLOB_DATA,
};
use crate::base::scratch::Scratch;
use core::convert::TryInto;
use core::mem::{align_of, size_of};

#[repr(C)]
pub(crate) struct BlobMeta {
    pub(crate) name_offset: usize,
    pub(crate) name_len: usize,
    pub(crate) dtype: u8,
    pub(crate) ndim: u8,
    pub(crate) shape_offset: usize,
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) digest_sha256: [u8; 32],
    pub(crate) has_stored_digest: bool,
    pub(crate) orig_length: u64,
    pub(crate) is_compressed: bool,
}

pub(crate) struct RnnHandle<'bytes, 'scratch> {
    pub(crate) bytes: &'bytes [u8],
    pub(crate) blobs: &'scratch [BlobMeta],
    pub(crate) scratch: &'scratch [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Error {
    Truncated,
    BadMagic,
    WrongFormatRmd1,
    BadHeader,
    BadBounds,
    ScratchFull,
}

const MAX_HEADER: usize = 65536;

fn blob_name_from_compact_id(id: u8) -> Option<&'static str> {
    match id {
        1 => Some(BLOB_NEURON_POSITIONS),
        2 => Some(BLOB_LAYER_META),
        3 => Some(BLOB_WEIGHTS),
        4 => Some(BLOB_BIASES),
        5 => Some(BLOB_RUNTIME_INPUT),
        6 => Some(BLOB_TRAINING_LOG),
        7 => Some(LMLP_CONFIG_BLOB_DATA),
        8 => Some(TOKENIZER_VOCAB_BLOB_DATA),
        9 => Some(TOKENIZER_MERGES_BLOB_DATA),
        10 => Some(BLOB_CONV_SPEC),
        11 => Some(LMLP_WEIGHTS_BLOB_DATA),
        12 => Some(TRAIN_CONFIG_BLOB_DATA),
        13 => Some(crate::format::model_config::ingest::MODEL_PRECISION_BLOB_DATA),
        14 => Some(LMLP_GRAPH_BLOB_DATA),
        15 => Some(LMLP_TENSORS_BLOB_DATA),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RnnContainerFormat {
    Rnn0,
}

fn magic_for_rnn_format(fmt: RnnContainerFormat) -> [u8; 4] {
    match fmt {
        RnnContainerFormat::Rnn0 => *b"RNN\x00",
    }
}

fn is_rmd1_magic(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && constant_time_eq(&bytes[0..4], b"RMD1")
}

const MAX_VIZ_ENTRIES: usize = 16;
const MAX_VIZ_NAME: usize = 32;

struct VizCompressInfo {
    count: usize,
    name_lens: [u8; MAX_VIZ_ENTRIES],
    names: [[u8; MAX_VIZ_NAME]; MAX_VIZ_ENTRIES],
    orig_sizes: [u64; MAX_VIZ_ENTRIES],
    stored_sizes: [u64; MAX_VIZ_ENTRIES],
    flags: [u8; MAX_VIZ_ENTRIES],
}

impl VizCompressInfo {
    fn empty() -> Self {
        VizCompressInfo {
            count: 0,
            name_lens: [0u8; MAX_VIZ_ENTRIES],
            names: [[0u8; MAX_VIZ_NAME]; MAX_VIZ_ENTRIES],
            orig_sizes: [0u64; MAX_VIZ_ENTRIES],
            stored_sizes: [0u64; MAX_VIZ_ENTRIES],
            flags: [0u8; MAX_VIZ_ENTRIES],
        }
    }

    fn lookup(&self, name: &[u8]) -> (u64, bool) {
        for i in 0..self.count {
            let nlen = self.name_lens[i] as usize;
            if nlen == name.len() && &self.names[i][..nlen] == name {
                let is_compressed = (self.flags[i] & 1) != 0;
                let orig = if is_compressed { self.orig_sizes[i] } else { self.stored_sizes[i] };
                return (orig, is_compressed);
            }
        }
        (0, false)
    }
}

fn scan_viz_info(bytes: &[u8], header_size: usize) -> VizCompressInfo {
    let mut info = VizCompressInfo::empty();
    let mut cursor = 12usize;
    while cursor + 5 <= header_size {
        if bytes[cursor] == 0 {
            cursor += 1;
            continue;
        }
        let t = bytes[cursor];
        cursor += 1;
        if cursor + 4 > header_size {
            break;
        }
        let l = u32::from_le_bytes([bytes[cursor], bytes[cursor+1], bytes[cursor+2], bytes[cursor+3]]) as usize;
        cursor += 4;
        if cursor + l > header_size {
            break;
        }
        if t == crate::format::model_format::TLV_VIZ_INDEX {
            let v = &bytes[cursor..cursor+l];
            if l >= 6 && &v[0..4] == b"VIZX" {
                let blob_count = v[5] as usize;
                let mut p = 6usize;
                let mut i = 0usize;
                while i < blob_count && i < MAX_VIZ_ENTRIES {
                    if p >= v.len() { break; }
                    let nlen = v[p] as usize;
                    p += 1;
                    if p + nlen + 25 > v.len() { break; }
                    let name_copy_len = nlen.min(MAX_VIZ_NAME);
                    info.name_lens[i] = name_copy_len as u8;
                    info.names[i][..name_copy_len].copy_from_slice(&v[p..p+name_copy_len]);
                    p += nlen;
                    let file_offset = u64::from_le_bytes([v[p],v[p+1],v[p+2],v[p+3],v[p+4],v[p+5],v[p+6],v[p+7]]);
                    p += 8;
                    let orig_size = u64::from_le_bytes([v[p],v[p+1],v[p+2],v[p+3],v[p+4],v[p+5],v[p+6],v[p+7]]);
                    p += 8;
                    let stored_size = u64::from_le_bytes([v[p],v[p+1],v[p+2],v[p+3],v[p+4],v[p+5],v[p+6],v[p+7]]);
                    p += 8;
                    let flags = v[p];
                    p += 1;
                    core::hint::black_box(file_offset);
                    info.orig_sizes[i] = orig_size;
                    info.stored_sizes[i] = stored_size;
                    info.flags[i] = flags;
                    i += 1;
                }
                info.count = i;
            }
        }
        cursor = match cursor.checked_add(l) { Some(v) => v, None => break };
    }
    info
}

pub(crate) fn parse_rnn_from_bytes<'bytes, 'scratch>(
    bytes: &'bytes [u8],
    scratch: &'scratch mut Scratch<'_>,
) -> Result<RnnHandle<'bytes, 'scratch>, Error> {
    parse_rnn_from_bytes_with_format(bytes, scratch, RnnContainerFormat::Rnn0)
}

pub(crate) fn parse_rnn_from_bytes_with_format<'bytes, 'scratch>(
    bytes: &'bytes [u8],
    scratch: &'scratch mut Scratch<'_>,
    format: RnnContainerFormat,
) -> Result<RnnHandle<'bytes, 'scratch>, Error> {
    if bytes.len() < 12 {
        return Err(Error::Truncated);
    }
    let expected_magic = magic_for_rnn_format(format);
    if !constant_time_eq(&bytes[0..4], &expected_magic) {
        if is_rmd1_magic(bytes) {
            return Err(Error::WrongFormatRmd1);
        }
        return Err(Error::BadMagic);
    }
    let version = u16::from_le_bytes(bytes[4..6].try_into().map_err(|_| Error::BadHeader)?);
    let flags = u16::from_le_bytes(bytes[6..8].try_into().map_err(|_| Error::BadHeader)?);
    core::hint::black_box(version);
    core::hint::black_box(flags);
    let header_size =
        u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| Error::BadHeader)?) as usize;
    if header_size > bytes.len() || header_size > MAX_HEADER {
        return Err(Error::BadHeader);
    }
    let viz_info = scan_viz_info(bytes, header_size);
    let scratch_base = scratch.base_ptr() as usize;

    if header_size >= crate::format::model_config::CANONICAL_HEADER_SIZE && bytes.len() >= crate::format::model_config::CANONICAL_HEADER_SIZE {
        let tbl_off = bytes[0x0E] as usize;
        if tbl_off < crate::format::model_config::CANONICAL_HEADER_SIZE && bytes[tbl_off] == 0xC1 {
        let table = &bytes[tbl_off..crate::format::model_config::CANONICAL_HEADER_SIZE];
        if table[1] != 1 {
            return Err(Error::BadHeader);
        }
        let count = table[2] as usize;
        let need = 3usize
            .checked_add(count.checked_mul(15).ok_or(Error::BadHeader)?)
            .ok_or(Error::BadHeader)?;
        if need > table.len() {
            return Err(Error::BadHeader);
        }

        let blobs_slice: &[BlobMeta] = if count == 0 {
            &[]
        } else {
            let meta_bytes = count
                .checked_mul(size_of::<BlobMeta>())
                .ok_or(Error::ScratchFull)?;
            let meta_store = scratch
                .alloc_align(meta_bytes, align_of::<BlobMeta>())
                .ok_or(Error::ScratchFull)?;
            let meta_ptr = meta_store.as_mut_ptr() as *mut BlobMeta;

            for i in 0..count {
                let p = 3 + i * 15;
                let blob_id = table[p];
                let dtype = table[p + 1];
                let ndim = table[p + 2];
                if !matches!(dtype, 0..=2) || ndim == 0 || ndim > 2 {
                    return Err(Error::BadHeader);
                }
                let d0 = u16::from_le_bytes([table[p + 3], table[p + 4]]) as u32;
                let d1 = u16::from_le_bytes([table[p + 5], table[p + 6]]) as u32;
                let offset = u32::from_le_bytes([
                    table[p + 7],
                    table[p + 8],
                    table[p + 9],
                    table[p + 10],
                ]) as u64;
                let length = u32::from_le_bytes([
                    table[p + 11],
                    table[p + 12],
                    table[p + 13],
                    table[p + 14],
                ]) as u64;

                let name = blob_name_from_compact_id(blob_id).ok_or(Error::BadHeader)?;
                let name_bytes = name.as_bytes();
                let name_len = name_bytes.len();
                let name_rel = {
                    let name_store = scratch
                        .alloc_align(name_len, align_of::<u8>())
                        .ok_or(Error::ScratchFull)?;
                    name_store.copy_from_slice(name_bytes);
                    name_store.as_ptr() as usize - scratch_base
                };

                let shape_bytes = (ndim as usize).saturating_mul(4);
                let dims_rel = {
                    let dims_store = scratch
                        .alloc_align(shape_bytes, align_of::<u32>())
                        .ok_or(Error::ScratchFull)?;
                    let mut w = 0usize;
                    if ndim >= 1 {
                        dims_store[w..w + 4].copy_from_slice(&d0.to_le_bytes());
                        w += 4;
                    }
                    if ndim >= 2 {
                        dims_store[w..w + 4].copy_from_slice(&d1.to_le_bytes());
                    }
                    dims_store.as_ptr() as usize - scratch_base
                };

                let offset_usize = usize::try_from(offset).map_err(|_| Error::BadBounds)?;
                let length_usize = usize::try_from(length).map_err(|_| Error::BadBounds)?;
                let end = offset_usize
                    .checked_add(length_usize)
                    .ok_or(Error::BadBounds)?;
                if end > bytes.len() {
                    return Err(Error::BadBounds);
                }
                let digest_sha256 = [0u8; 32];
                let (orig_length, is_compressed) = viz_info.lookup(name_bytes);
                let orig_length = if orig_length == 0 { length } else { orig_length };

                unsafe {
                    core::ptr::write(
                        meta_ptr.add(i),
                        BlobMeta {
                            name_offset: name_rel,
                            name_len,
                            dtype,
                            ndim,
                            shape_offset: dims_rel,
                            offset,
                            length,
                            digest_sha256,
                            has_stored_digest: false,
                            orig_length,
                            is_compressed,
                        },
                    );
                }
            }

            unsafe { core::slice::from_raw_parts(meta_ptr as *const BlobMeta, count) }
        };

        return Ok(RnnHandle {
            bytes,
            blobs: blobs_slice,
            scratch: scratch.as_slice(),
        });
    }
    }

    let mut cursor = 12usize;

    let mut metas_count = 0usize;

    while cursor < header_size {
        if bytes[cursor] == 0 {
            cursor += 1;
            continue;
        }
        if cursor + 5 > header_size {
            return Err(Error::BadHeader);
        }
        let t = bytes[cursor];
        cursor += 1;
        let l = u32::from_le_bytes(
            bytes[cursor..cursor + 4]
                .try_into()
                .map_err(|_| Error::BadHeader)?,
        ) as usize;
        cursor += 4;
        if cursor + l > header_size {
            return Err(Error::BadHeader);
        }
        if t == 0x03 {
            if l >= 3 && bytes[cursor] == 0xC1 {
                let version_compact = bytes[cursor + 1];
                if version_compact != 1 {
                    return Err(Error::BadHeader);
                }
                let count = bytes[cursor + 2] as usize;
                let need = 3usize
                    .checked_add(count.checked_mul(15).ok_or(Error::BadHeader)?)
                    .ok_or(Error::BadHeader)?;
                if need > l {
                    return Err(Error::BadHeader);
                }
                let mut p = cursor + 3;
                for _ in 0..count {
                    let blob_id = bytes[p];
                    let dtype = bytes[p + 1];
                    let ndim = bytes[p + 2];
                    let d0 = u16::from_le_bytes([bytes[p + 3], bytes[p + 4]]) as u32;
                    let d1 = u16::from_le_bytes([bytes[p + 5], bytes[p + 6]]) as u32;
                    let offset = u32::from_le_bytes([
                        bytes[p + 7],
                        bytes[p + 8],
                        bytes[p + 9],
                        bytes[p + 10],
                    ]) as u64;
                    let length = u32::from_le_bytes([
                        bytes[p + 11],
                        bytes[p + 12],
                        bytes[p + 13],
                        bytes[p + 14],
                    ]) as u64;
                    p += 15;

                    if blob_name_from_compact_id(blob_id).is_none() {
                        return Err(Error::BadHeader);
                    }
                    if !matches!(dtype, 0..=2) || ndim == 0 || ndim > 2 {
                        return Err(Error::BadHeader);
                    }
                    let _ = (d0, d1);
                    let offset_usize = usize::try_from(offset).map_err(|_| Error::BadBounds)?;
                    let length_usize = usize::try_from(length).map_err(|_| Error::BadBounds)?;
                    let end = offset_usize
                        .checked_add(length_usize)
                        .ok_or(Error::BadBounds)?;
                    if end > bytes.len() {
                        return Err(Error::BadBounds);
                    }
                    metas_count = metas_count.checked_add(1).ok_or(Error::ScratchFull)?;
                }
                cursor = cursor.checked_add(l).ok_or(Error::BadHeader)?;
                continue;
            }

            let mut p = cursor;
            while p < cursor + l {
                if p + 2 > cursor + l {
                    return Err(Error::BadHeader);
                }
                let name_len =
                    u16::from_le_bytes(bytes[p..p + 2].try_into().map_err(|_| Error::BadHeader)?)
                        as usize;
                p += 2;
                if p + name_len + 1 + 1 > cursor + l {
                    return Err(Error::BadHeader);
                }
                p += name_len;
                if !matches!(bytes[p], 0..=2) {
                    return Err(Error::BadHeader);
                }
                p += 1;
                let ndim = bytes[p];
                p += 1;
                if ndim == 0 {
                    return Err(Error::BadHeader);
                }
                let shape_bytes = (ndim as usize).saturating_mul(4);
                if p + shape_bytes + 8 + 8 + 32 > cursor + l {
                    return Err(Error::BadHeader);
                }
                p += shape_bytes;
                let offset =
                    u64::from_le_bytes(bytes[p..p + 8].try_into().map_err(|_| Error::BadHeader)?);
                p += 8;
                let length =
                    u64::from_le_bytes(bytes[p..p + 8].try_into().map_err(|_| Error::BadHeader)?);
                p += 8;
                p += 32;
                let offset_usize = usize::try_from(offset).map_err(|_| Error::BadBounds)?;
                let length_usize = usize::try_from(length).map_err(|_| Error::BadBounds)?;
                let end = offset_usize
                    .checked_add(length_usize)
                    .ok_or(Error::BadBounds)?;
                if end > bytes.len() {
                    return Err(Error::BadBounds);
                }
                metas_count = metas_count.checked_add(1).ok_or(Error::ScratchFull)?;
            }
        }
        cursor = cursor.checked_add(l).ok_or(Error::BadHeader)?;
    }

    let blobs_slice: &[BlobMeta] = if metas_count == 0 {
        &[]
    } else {
        let meta_bytes = metas_count
            .checked_mul(size_of::<BlobMeta>())
            .ok_or(Error::ScratchFull)?;
        let meta_store = scratch
            .alloc_align(meta_bytes, align_of::<BlobMeta>())
            .ok_or(Error::ScratchFull)?;
        let meta_ptr = meta_store.as_mut_ptr() as *mut BlobMeta;

        cursor = 12usize;
        let mut meta_index = 0usize;

        while cursor < header_size {
            if bytes[cursor] == 0 {
                cursor += 1;
                continue;
            }
            if cursor + 5 > header_size {
                return Err(Error::BadHeader);
            }
            let t = bytes[cursor];
            cursor += 1;
            let l = u32::from_le_bytes(
                bytes[cursor..cursor + 4]
                    .try_into()
                    .map_err(|_| Error::BadHeader)?,
            ) as usize;
            cursor += 4;
            if cursor + l > header_size {
                return Err(Error::BadHeader);
            }

            if t == 0x03 {
                if l >= 3 && bytes[cursor] == 0xC1 {
                    let version_compact = bytes[cursor + 1];
                    if version_compact != 1 {
                        return Err(Error::BadHeader);
                    }
                    let count = bytes[cursor + 2] as usize;
                    let need = 3usize
                        .checked_add(count.checked_mul(15).ok_or(Error::BadHeader)?)
                        .ok_or(Error::BadHeader)?;
                    if need > l {
                        return Err(Error::BadHeader);
                    }
                    let mut p = cursor + 3;
                    for _ in 0..count {
                        let blob_id = bytes[p];
                        let dtype = bytes[p + 1];
                        let ndim = bytes[p + 2];
                        if blob_name_from_compact_id(blob_id).is_none() {
                            return Err(Error::BadHeader);
                        }
                        if !matches!(dtype, 0..=2) || ndim == 0 || ndim > 2 {
                            return Err(Error::BadHeader);
                        }
                        let d0 = u16::from_le_bytes([bytes[p + 3], bytes[p + 4]]) as u32;
                        let d1 = u16::from_le_bytes([bytes[p + 5], bytes[p + 6]]) as u32;
                        let offset = u32::from_le_bytes([
                            bytes[p + 7],
                            bytes[p + 8],
                            bytes[p + 9],
                            bytes[p + 10],
                        ]) as u64;
                        let length = u32::from_le_bytes([
                            bytes[p + 11],
                            bytes[p + 12],
                            bytes[p + 13],
                            bytes[p + 14],
                        ]) as u64;
                        p += 15;

                        let offset_usize = usize::try_from(offset).map_err(|_| Error::BadBounds)?;
                        let length_usize = usize::try_from(length).map_err(|_| Error::BadBounds)?;
                        let end = offset_usize
                            .checked_add(length_usize)
                            .ok_or(Error::BadBounds)?;
                        if end > bytes.len() {
                            return Err(Error::BadBounds);
                        }
                        let digest_sha256 = [0u8; 32];

                        let name = blob_name_from_compact_id(blob_id).ok_or(Error::BadHeader)?;
                        let name_bytes = name.as_bytes();
                        let name_len = name_bytes.len();
                        let name_rel = {
                            let name_store = scratch
                                .alloc_align(name_len, align_of::<u8>())
                                .ok_or(Error::ScratchFull)?;
                            name_store.copy_from_slice(name_bytes);
                            name_store.as_ptr() as usize - scratch_base
                        };

                        let shape_bytes = (ndim as usize).saturating_mul(4);
                        let dims_rel = {
                            let dims_store = scratch
                                .alloc_align(shape_bytes, align_of::<u32>())
                                .ok_or(Error::ScratchFull)?;
                            let mut w = 0usize;
                            if ndim >= 1 {
                                dims_store[w..w + 4].copy_from_slice(&d0.to_le_bytes());
                                w += 4;
                            }
                            if ndim >= 2 {
                                dims_store[w..w + 4].copy_from_slice(&d1.to_le_bytes());
                            }
                            dims_store.as_ptr() as usize - scratch_base
                        };

                        let (orig_length, is_compressed) = viz_info.lookup(name_bytes);
                        let orig_length = if orig_length == 0 { length } else { orig_length };

                        unsafe {
                            core::ptr::write(
                                meta_ptr.add(meta_index),
                                BlobMeta {
                                    name_offset: name_rel,
                                    name_len,
                                    dtype,
                                    ndim,
                                    shape_offset: dims_rel,
                                    offset,
                                    length,
                                    digest_sha256,
                                    has_stored_digest: false,
                                    orig_length,
                                    is_compressed,
                                },
                            );
                        }
                        meta_index = meta_index.checked_add(1).ok_or(Error::ScratchFull)?;
                    }

                    cursor = cursor.checked_add(l).ok_or(Error::BadHeader)?;
                    continue;
                }

                let mut p = cursor;
                while p < cursor + l {
                    if p + 2 > cursor + l {
                        return Err(Error::BadHeader);
                    }
                    let name_len = u16::from_le_bytes(
                        bytes[p..p + 2].try_into().map_err(|_| Error::BadHeader)?,
                    ) as usize;
                    p += 2;
                    if p + name_len + 1 + 1 > cursor + l {
                        return Err(Error::BadHeader);
                    }
                    let name_bytes = &bytes[p..p + name_len];
                    p += name_len;
                    let dtype = bytes[p];
                    p += 1;
                    let ndim = bytes[p];
                    p += 1;
                    if ndim == 0 {
                        return Err(Error::BadHeader);
                    }
                    let shape_bytes = (ndim as usize).saturating_mul(4);
                    if p + shape_bytes + 8 + 8 + 32 > cursor + l {
                        return Err(Error::BadHeader);
                    }
                    let dims_src = &bytes[p..p + shape_bytes];
                    p += shape_bytes;
                    let offset = u64::from_le_bytes(
                        bytes[p..p + 8].try_into().map_err(|_| Error::BadHeader)?,
                    );
                    p += 8;
                    let length = u64::from_le_bytes(
                        bytes[p..p + 8].try_into().map_err(|_| Error::BadHeader)?,
                    );
                    p += 8;
                    let mut digest_sha256 = [0u8; 32];
                    digest_sha256.copy_from_slice(&bytes[p..p + 32]);
                    p += 32;
                    let digest_present = digest_sha256.iter().any(|byte| *byte != 0);

                    let offset_usize = usize::try_from(offset).map_err(|_| Error::BadBounds)?;
                    let length_usize = usize::try_from(length).map_err(|_| Error::BadBounds)?;
                    let end = offset_usize
                        .checked_add(length_usize)
                        .ok_or(Error::BadBounds)?;
                    if end > bytes.len() {
                        return Err(Error::BadBounds);
                    }

                    let name_rel = {
                        let name_store = scratch
                            .alloc_align(name_len, align_of::<u8>())
                            .ok_or(Error::ScratchFull)?;
                        name_store.copy_from_slice(name_bytes);
                        name_store.as_ptr() as usize - scratch_base
                    };

                    let dims_rel = {
                        let dims_store = scratch
                            .alloc_align(shape_bytes, align_of::<u32>())
                            .ok_or(Error::ScratchFull)?;
                        dims_store.copy_from_slice(dims_src);
                        dims_store.as_ptr() as usize - scratch_base
                    };

                    let (orig_length, is_compressed) = viz_info.lookup(name_bytes);
                    let orig_length = if orig_length == 0 { length } else { orig_length };

                    unsafe {
                        core::ptr::write(
                            meta_ptr.add(meta_index),
                            BlobMeta {
                                name_offset: name_rel,
                                name_len,
                                dtype,
                                ndim,
                                shape_offset: dims_rel,
                                offset,
                                length,
                                digest_sha256,
                                has_stored_digest: digest_present,
                                orig_length,
                                is_compressed,
                            },
                        );
                    }
                    meta_index = meta_index.checked_add(1).ok_or(Error::ScratchFull)?;
                }
            }
            cursor = cursor.checked_add(l).ok_or(Error::BadHeader)?;
        }

        if meta_index != metas_count {
            return Err(Error::BadHeader);
        }

        unsafe { core::slice::from_raw_parts(meta_ptr as *const BlobMeta, metas_count) }
    };

    Ok(RnnHandle {
        bytes,
        blobs: blobs_slice,
        scratch: scratch.as_slice(),
    })
}

impl<'bytes, 'scratch> RnnHandle<'bytes, 'scratch> {
    pub(crate) fn blob_name(&self, i: usize) -> Option<&'scratch str> {
        let m = self.blobs.get(i)?;
        let end = m.name_offset.checked_add(m.name_len)?;
        if end > self.scratch.len() {
            return None;
        }
        let sl = &self.scratch[m.name_offset..end];
        core::str::from_utf8(sl).ok()
    }
}
