#![deny(unused_must_use)]

pub mod error;
pub mod losses;
pub mod metrics;

pub use losses::{
    Reduction, binary_cross_entropy, binary_cross_entropy_with_logits, cross_entropy,
    cross_entropy_from_logits, cross_entropy_from_logits_sparse, mae, mse,
};
pub use metrics::{accuracy_top1, accuracy_topk};
