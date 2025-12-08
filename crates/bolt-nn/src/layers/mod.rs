mod activations;
mod linear;
mod sequential;

pub use activations::{relu, ReLU};
pub use linear::{linear, Linear, LinearSpec};
pub use sequential::{ModelExt, Seq, Then};
