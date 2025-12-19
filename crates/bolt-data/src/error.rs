use thiserror::Error;

#[derive(Debug, Error)]
pub enum DataError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("core error: {0}")]
    Core(#[from] bolt_core::Error),

    #[error("invalid magic number: expected {expected}, got {actual}")]
    InvalidMagic { expected: u32, actual: u32 },

    #[error("length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("shape error: {0}")]
    InvalidShape(String),

    #[error("index out of bounds: index {index}, len {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("empty batch")]
    EmptyBatch,

    #[error("batch construction error: {0}")]
    Batch(String),
}

pub type Result<T> = std::result::Result<T, DataError>;
