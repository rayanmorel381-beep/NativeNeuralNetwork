#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerError {
    UnknownByte,
    BufferTooSmall,
    InvalidUtf8,
    VocabEmpty,
    MergeCapacityExceeded,
    BlobTruncated,
    BlobBadMagic,
    BlobBadVersion,
    BlobEntriesBufferTooSmall,
    TrainerNotEnoughBytes,
    TrainerCapacityExceeded,
}
