//! Neural network helpers built on Bolt backends.
//! Losses and metrics are re-exported from `bolt-losses` when the `losses`
//! feature is enabled (default).

mod context;
pub mod error;
pub mod layers;
mod model;

#[cfg(feature = "losses")]
pub use bolt_losses::*;
pub use context::{Context, Mode, Rng};
pub use error::{Error, Result};
pub use model::Model;
