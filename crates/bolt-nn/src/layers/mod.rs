mod activations;
mod linear;
mod sequential;

pub use activations::{ReLU, relu};
pub use bolt_autodiff::HasParams;
pub use linear::{Linear, LinearSpec, linear};
pub use sequential::{ModelExt, Seq, Then};
