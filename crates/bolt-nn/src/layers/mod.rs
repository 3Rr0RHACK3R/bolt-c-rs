mod activations;
mod flatten;
mod linear;
mod sequential;

pub use activations::{ReLU, relu};
pub use bolt_autodiff::HasParams;
pub use flatten::{Flatten, flatten};
pub use linear::{Linear, LinearSpec, linear};
pub use sequential::{ModelExt, Seq, Then};
