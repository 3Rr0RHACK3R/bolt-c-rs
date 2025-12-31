use std::path::PathBuf;

use bolt_core::DType;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid record name '{name}': {reason}")]
    InvalidName { name: String, reason: String },

    #[error("duplicate record name: '{name}'")]
    DuplicateName { name: String },

    #[error(
        "record '{name}' byte size mismatch: expected {expected} bytes (numel={numel}, dtype={dtype:?}), got {actual}"
    )]
    ByteSizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
        numel: u64,
        dtype: DType,
    },

    #[error(
        "record '{name}' has invalid byte size: {actual} bytes is not a multiple of element size {element_size} for dtype {dtype:?}"
    )]
    ByteSizeNotAligned {
        name: String,
        actual: u64,
        element_size: usize,
        dtype: DType,
    },

    #[error("record '{name}' not found in artifact at {dir:?}")]
    RecordNotFound { name: String, dir: PathBuf },

    #[error("dtype mismatch for record '{name}': expected {expected:?}, found {found:?}")]
    DTypeMismatch {
        name: String,
        expected: DType,
        found: DType,
    },

    #[error("shape mismatch for record '{name}': expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error(
        "unsafe path detected: '{path}' (reason: {reason}). Try using a relative path within the artifact directory."
    )]
    UnsafePath { path: String, reason: String },

    #[error("checksum mismatch for shard '{shard}': expected {expected}, computed {computed}")]
    ShardChecksumMismatch {
        shard: PathBuf,
        expected: String,
        computed: String,
    },

    #[error("record '{name}' in shard '{shard}' is corrupted (checksum mismatch)")]
    RecordCorrupted { name: String, shard: PathBuf },

    #[error("schema version mismatch in '{file}': expected '{expected}', found '{found}'")]
    SchemaVersionMismatch {
        file: PathBuf,
        expected: String,
        found: String,
    },

    #[error("artifact directory already exists: {dir:?}. Set overwrite=true to replace.")]
    DirectoryExists { dir: PathBuf },

    #[error("artifact directory not found: {dir:?}")]
    DirectoryNotFound { dir: PathBuf },

    #[error("manifest file not found: {path:?}")]
    ManifestNotFound { path: PathBuf },

    #[error("shard file not found: {path:?}")]
    ShardNotFound { path: PathBuf },

    #[error("failed to parse manifest at {path:?}: {reason}")]
    ManifestParse { path: PathBuf, reason: String },

    #[error("shard format error for '{shard:?}': {reason}")]
    ShardFormat { shard: PathBuf, reason: String },

    #[error("io error at '{path:?}': {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("atomic save failed: could not rename temp directory to final")]
    AtomicRenameFailed {
        temp: PathBuf,
        final_dir: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("numel overflow: shape {shape:?} exceeds maximum representable size")]
    NumelOverflow { shape: Vec<usize> },

    #[error("record '{name}' is unavailable due to shard corruption.")]
    RecordUnavailable { name: String },

    #[error(
        "internal inconsistency: shape data missing for record '{name}' in artifact at {dir:?}. This indicates a bug or data corruption."
    )]
    ShapeMissing { name: String, dir: PathBuf },

    #[error("invalid exclude pattern '{pattern}': {source}")]
    InvalidExcludePattern {
        pattern: String,
        #[source]
        source: glob::PatternError,
    },

    #[error("failed to materialize tensor '{name}': {reason}")]
    TensorMaterializeFailed { name: String, reason: String },

    #[error("failed to restore tensor '{name}': {reason}")]
    TensorRestoreFailed { name: String, reason: String },

    #[error("{reason}")]
    RestoreFailed { reason: String },
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
