mod activations;
mod linear;
mod sequential;

pub use activations::{ReLU, relu};
pub use linear::{Linear, LinearSpec, linear};
pub use sequential::{ModelExt, Seq, Then};
