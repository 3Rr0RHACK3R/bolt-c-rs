use crate::Handle;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid handle: {reason}")]
    InvalidHandle { reason: String },

    #[error("tensor not found for handle {handle:?}")]
    TensorNotFound { handle: Handle },

    #[error("operation requires leaf tensor: {reason}")]
    NotALeaf { reason: String },

    #[error("cannot compute backward: loss tensor does not require gradient")]
    LossNoGrad,

    #[error("cannot compute backward: gradient computation is disabled")]
    GradDisabled,

    #[error("backward op not found at index {idx}")]
    BackwardOpNotFound { idx: usize },

    #[error("saved tensors not found at index {idx}")]
    SavedTensorsNotFound { idx: usize },

    #[error(transparent)]
    Core(#[from] bolt_core::Error),
}

impl Error {
    pub fn stale_handle() -> Self {
        Self::InvalidHandle {
            reason: "handle is from a previous graph generation (call to clear() invalidated it)"
                .into(),
        }
    }

    pub fn handle_out_of_bounds(idx: u32, len: usize) -> Self {
        Self::InvalidHandle {
            reason: format!("handle index {} out of bounds (graph has {} nodes)", idx, len),
        }
    }
}
