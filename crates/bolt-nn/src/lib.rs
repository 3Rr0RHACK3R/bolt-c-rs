//! Neural network helpers built on Bolt backends.
//! Losses and metrics are re-exported from `bolt-losses` when the `losses`
//! feature is enabled (default).

mod context;
pub mod error;
pub mod init;
pub mod layers;
pub mod run_mode;
pub mod dual;
pub mod state_dict;
pub mod visit;
mod mode;
mod model;

pub use bolt_autodiff::HasParams;
#[cfg(feature = "losses")]
pub use bolt_losses::*;
pub use context::{Context, Rng};
pub use error::{Error, Result};
pub use mode::{Eval, Grad, Mode};
pub use model::Model;
pub use run_mode::{RunMode, Trainable};
pub use dual::DualModel;
