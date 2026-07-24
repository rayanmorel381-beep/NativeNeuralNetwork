const TRAIN_CONFIG_MAGIC: &[u8; 4] = b"TCF1";

pub(crate) struct TrainConfigView<'a> {
    pub train_window: usize,
    pub seed: u64,
    pub dir_count: usize,
    blob: &'a [u8],
    dirs_off: usize,
}

impl<'a> TrainConfigView<'a> {
    pub(crate) fn dir(&self, index: usize) -> Option<&'a str> {
        let mut cursor = self.dirs_off;
        for i in 0..self.dir_count {
            if cursor + 4 > self.blob.len() {
                return None;
            }
            let len = u32::from_le_bytes([
                self.blob[cursor],
                self.blob[cursor + 1],
                self.blob[cursor + 2],
                self.blob[cursor + 3],
            ]) as usize;
            cursor += 4;
            if cursor + len > self.blob.len() {
                return None;
            }
            if i == index {
                return core::str::from_utf8(&self.blob[cursor..cursor + len]).ok();
            }
            cursor += len;
        }
        None
    }
}

pub(crate) fn train_config_encode(
    train_window: usize,
    seed: u64,
    dataset_dirs: &[&str],
    out: &mut [u8],
) -> Option<usize> {
    let mut cursor = 0usize;
    let mut put = |slice: &[u8], cursor: &mut usize| -> bool {
        if *cursor + slice.len() > out.len() {
            return false;
        }
        out[*cursor..*cursor + slice.len()].copy_from_slice(slice);
        *cursor += slice.len();
        true
    };
    if !put(TRAIN_CONFIG_MAGIC, &mut cursor) {
        return None;
    }
    if !put(&(train_window as u32).to_le_bytes(), &mut cursor) {
        return None;
    }
    if !put(&seed.to_le_bytes(), &mut cursor) {
        return None;
    }
    if !put(&(dataset_dirs.len() as u32).to_le_bytes(), &mut cursor) {
        return None;
    }
    for dir in dataset_dirs {
        let bytes = dir.as_bytes();
        if !put(&(bytes.len() as u32).to_le_bytes(), &mut cursor) {
            return None;
        }
        if !put(bytes, &mut cursor) {
            return None;
        }
    }
    Some(cursor)
}

pub(crate) fn train_config_decode(blob: &[u8]) -> Option<TrainConfigView<'_>> {
    if blob.len() < 20 || &blob[0..4] != TRAIN_CONFIG_MAGIC {
        return None;
    }
    let train_window = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]) as usize;
    let seed = u64::from_le_bytes([
        blob[8], blob[9], blob[10], blob[11], blob[12], blob[13], blob[14], blob[15],
    ]);
    let dir_count = u32::from_le_bytes([blob[16], blob[17], blob[18], blob[19]]) as usize;
    Some(TrainConfigView { train_window, seed, dir_count, blob, dirs_off: 20 })
}
