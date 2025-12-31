#![deny(unused_must_use)]

mod error;
mod forward_ctx;
mod init;
mod module;
mod store;

pub mod layers;

pub use error::{Error, Result};
pub use forward_ctx::ForwardCtx;
pub use init::{Init, fill};
pub use module::Module;
pub use store::{Buffer, Kind, Param, Store};
