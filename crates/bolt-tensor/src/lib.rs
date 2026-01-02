#![deny(unused_must_use)]

pub mod autograd;
mod display;
pub mod tensor;
pub(crate) mod utils;

pub use autograd::{BackwardOptions, Grads, NoGradGuard, backward, backward_with_options, grad_enabled, no_grad};
pub use tensor::{Tensor, ToBackend};
