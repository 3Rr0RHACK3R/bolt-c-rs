#![deny(unused_must_use)]

mod batch;
mod error;
mod idx;
mod source;
mod stream;

pub use crate::batch::BatchFromExamples;
pub use crate::error::{DataError, Result};
pub use crate::idx::{IdxExample, IdxSpec, IdxSplit, idx_dataset};
pub use crate::source::{
    BatchSource, EnumerateSource, MapSource, ShuffleSource, Source, TakeSource,
};
pub use crate::stream::{Stream, StreamIter};
