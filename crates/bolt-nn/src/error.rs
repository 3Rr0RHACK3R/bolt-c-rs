use bolt_autodiff::Error as AutodiffError;
use bolt_core::Error as CoreError;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("core error: {0}")]
    Core(#[from] CoreError),

    #[error("autodiff error: {0}")]
    Autodiff(#[from] AutodiffError),

    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("missing parameter: {0}")]
    MissingParam(String),

    #[error(
        "incompatible shared parameter: key '{key}' has shape {existing:?}, but {new:?} was requested"
    )]
    IncompatibleSharedParam {
        key: String,
        existing: Vec<usize>,
        new: Vec<usize>,
    },
}
