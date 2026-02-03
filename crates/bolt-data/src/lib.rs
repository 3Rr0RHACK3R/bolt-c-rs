#![deny(unused_must_use)]

mod error;
mod idx;
mod source;
mod stream;
mod stream_ext;

pub use crate::error::{DataError, Result};
pub use crate::idx::{IdxExample, IdxSpec, IdxSplit, idx_dataset};
pub use crate::source::{
    BatchSource, EnumerateSource, MapSource, MapWithSource, ShuffleSource, Source, TakeSource,
    TryMapSource, TryMapWithSource,
};
pub use crate::stream::{BatchRemainder, Stream, StreamIter};
