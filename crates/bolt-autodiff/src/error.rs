use crate::{Handle, parameter::ParamId};
use bolt_core::Error as CoreError;
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

    #[error("cannot compute backward: no active gradient context (use begin_grad())")]
    NoActiveGraph,

    #[error("parameter not recorded in this tape: id={param_id:?}, name={param_name:?}")]
    ParamNotInTape {
        param_id: ParamId,
        param_name: Option<String>,
    },

    #[error("backward op not found at index {idx}")]
    BackwardOpNotFound { idx: usize },

    #[error("saved tensors not found at index {idx}")]
    SavedTensorsNotFound { idx: usize },

    #[error("Function::apply requires at least one input tensor")]
    EmptyInputs,

    #[error("Function::forward must return at least one output tensor")]
    EmptyOutputs,

    #[error(transparent)]
    Core(#[from] CoreError),
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
            reason: format!(
                "handle index {} out of bounds (graph has {} nodes)",
                idx, len
            ),
        }
    }
}
