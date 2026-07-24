use super::parser::RnnHandle;

pub(crate) fn find_blob_index(handle: &RnnHandle<'_, '_>, name: &str) -> Option<usize> {
    (0..handle.blobs.len()).find(|&i| handle.blob_name(i) == Some(name))
}
