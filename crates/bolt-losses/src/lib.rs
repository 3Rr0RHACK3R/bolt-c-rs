#![deny(unused_must_use)]

pub mod error;
pub mod losses;
pub mod metrics;

pub use error::{Error, Result};
pub use losses::{Reduction, cross_entropy, cross_entropy_from_logits, mse};
pub use metrics::accuracy_top1;
