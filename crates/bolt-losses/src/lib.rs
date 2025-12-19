#![deny(unused_must_use)]

pub mod error;
pub mod losses;
pub mod metrics;

pub use losses::{
    Reduction, cross_entropy, cross_entropy_from_logits, cross_entropy_from_logits_sparse, mse,
};
pub use metrics::accuracy_top1;
