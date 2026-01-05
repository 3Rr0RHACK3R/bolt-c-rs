use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Invalid checkpoint: {0}")]
    InvalidCheckpoint(String),

    #[error("Key not found: {0}")]
    KeyNotFound(String),

    #[error("Duplicate key: {0}")]
    DuplicateKey(String),

    #[error("Format error: {0}")]
    Format(String),

    #[error("Invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("Invalid dtype: expected {expected:?}, got {got:?}")]
    InvalidDtype { expected: String, got: String },
}

pub type Result<T> = std::result::Result<T, Error>;
