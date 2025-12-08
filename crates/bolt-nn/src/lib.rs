mod context;
pub mod error;
pub mod layers;
mod model;

pub use context::{Context, Mode, Rng};
pub use error::{Error, Result};
pub use model::Model;
