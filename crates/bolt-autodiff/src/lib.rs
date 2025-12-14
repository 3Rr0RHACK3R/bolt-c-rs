#![deny(unused_must_use)]

mod backend;
mod backward;
pub mod error;
mod function;
mod grad_tape;
mod gradients;
mod graph;
mod handle;
mod operations;
pub mod ops;
mod parameter;
mod scope;
mod storage;
mod tensor_ext;
mod utils;

pub use backward::{BackwardContext, BackwardOp, MAX_INPUTS};
pub use bolt_core::Float;
pub use error::{Error, Result};
pub use function::Function;
pub use grad_tape::{GradTape, ParamGrads};
pub use gradients::Gradients;
pub use handle::Handle;
pub use operations::Autodiff;
pub use parameter::{HasParams, ParamId, Parameter};
pub use scope::{GradContext, NoGradGuard};
pub use storage::{AutodiffAllocator, AutodiffStorage};
pub use tensor_ext::{AutodiffBackend, AutodiffTensorExt};
