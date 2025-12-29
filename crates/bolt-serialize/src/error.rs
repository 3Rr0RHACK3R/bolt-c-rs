use std::path::PathBuf;

use bolt_core::DType;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid tensor name '{name}': {reason}")]
    InvalidName { name: String, reason: String },

    #[error("duplicate tensor name: '{name}'")]
    DuplicateName { name: String },

    #[error(
        "tensor '{name}' byte size mismatch: expected {expected} bytes (numel={numel}, dtype={dtype:?}), got {actual}"
    )]
    ByteSizeMismatch {
        name: String,
        expected: u64,
        actual: u64,
        numel: u64,
        dtype: DType,
    },

    #[error("tensor '{name}' not found in artifact at {dir:?}")]
    TensorNotFound { name: String, dir: PathBuf },

    #[error("dtype mismatch for tensor '{name}': expected {expected:?}, found {found:?}")]
    DTypeMismatch {
        name: String,
        expected: DType,
        found: DType,
    },

    #[error("shape mismatch for tensor '{name}': expected {expected:?}, found {found:?}")]
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

    #[error("tensor '{name}' in shard '{shard}' is corrupted (checksum mismatch)")]
    TensorCorrupted { name: String, shard: PathBuf },

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

    #[error("safetensors error for shard '{shard:?}': {reason}")]
    Safetensors { shard: PathBuf, reason: String },

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

    #[error(
        "tensor '{name}' is unavailable due to shard corruption. Try loading with ErrorMode::Permissive to inspect remaining tensors."
    )]
    TensorUnavailable { name: String },

    #[error(
        "internal inconsistency: shape data missing for tensor '{name}' in artifact at {dir:?}. This indicates a bug or data corruption."
    )]
    ShapeMissing { name: String, dir: PathBuf },

    #[error("invalid exclude pattern '{pattern}': {source}")]
    InvalidExcludePattern {
        pattern: String,
        #[source]
        source: glob::PatternError,
    },
}

impl Error {
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
