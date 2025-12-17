use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("download failed: {0}")]
    Download(String),

    #[error("data error: {0}")]
    Data(#[from] bolt_data::DataError),
}

pub type Result<T> = std::result::Result<T, DatasetError>;
