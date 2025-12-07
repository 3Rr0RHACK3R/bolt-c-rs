#![deny(unused_must_use)]

mod backend;
mod backward;
pub mod device;
pub mod error;
mod float;
mod gradients;
mod graph;
mod handle;
mod operations;
pub mod ops;
mod scope;
mod storage;
mod tensor_ext;
mod utils;

pub use backward::{BackwardContext, BackwardOp};
pub use device::AutodiffDevice;
pub use error::{Error, Result};
pub use float::Float;
pub use gradients::Gradients;
pub use handle::Handle;
pub use operations::Autodiff;
pub use scope::{GradContext, NoGradGuard};
pub use storage::{AutodiffAllocator, AutodiffStorage};
pub use tensor_ext::{AutodiffBackend, AutodiffTensorExt};
