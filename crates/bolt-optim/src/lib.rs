#![deny(unused_must_use)]

pub mod error;
mod sgd;

pub use error::{Error, Result};
pub use sgd::{Sgd, SgdBuilder, SgdConfig, SgdState};
