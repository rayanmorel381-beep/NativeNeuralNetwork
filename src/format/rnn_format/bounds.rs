use super::parser::RnnHandle;
use crate::security::crypto::{constant_time_eq, sha256_bytes};

pub(crate) fn blob_bounds_valid(handle: &RnnHandle<'_, '_>, index: usize) -> bool {
    let b = match handle.blobs.get(index) {
        Some(v) => v,
        None => return false,
    };
    let offset = match usize::try_from(b.offset) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let len = match usize::try_from(b.length) {
        Ok(v) => v,
        Err(_) => return false,
    };
    match offset.checked_add(len) {
        Some(end) => end <= handle.bytes.len(),
        None => false,
    }
}

pub(crate) fn all_blob_bounds_valid(handle: &RnnHandle<'_, '_>) -> bool {
    for i in 0..handle.blobs.len() {
        if !blob_bounds_valid(handle, i) {
            return false;
        }
    }
    true
}

pub(crate) fn blob_digest_valid(handle: &RnnHandle<'_, '_>, index: usize) -> bool {
    let b = match handle.blobs.get(index) {
        Some(v) => v,
        None => return false,
    };
    if !b.has_stored_digest {
        return true;
    }
    let offset = match usize::try_from(b.offset) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let len = match usize::try_from(b.length) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let end = match offset.checked_add(len) {
        Some(v) => v,
        None => return false,
    };
    let payload = match handle.bytes.get(offset..end) {
        Some(v) => v,
        None => return false,
    };

    let mut computed = [0u8; 32];
    sha256_bytes(payload, &mut computed);
    constant_time_eq(&computed, &b.digest_sha256)
}

pub(crate) fn all_blob_digests_valid(handle: &RnnHandle<'_, '_>) -> bool {
    for i in 0..handle.blobs.len() {
        if !blob_digest_valid(handle, i) {
            return false;
        }
    }
    true
}
