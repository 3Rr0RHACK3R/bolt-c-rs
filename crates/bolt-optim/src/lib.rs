#![deny(unused_must_use)]

mod adam;
mod sgd;

pub use adam::{Adam, AdamCfg, AdamGroupCfg};
pub use sgd::{Sgd, SgdCfg, SgdGroupCfg};
