mod batch;
mod download;

use std::path::Path;

use bolt_data::{IdxExample, IdxSpec, IdxSplit, Stream, idx_dataset};

use crate::Result;

pub use batch::MnistBatch;
pub use download::ensure_downloaded;

pub const SPEC: IdxSpec = IdxSpec::new(28, 28, 1);
pub const NUM_CLASSES: usize = 10;
pub const INPUT_DIM: usize = 28 * 28;

pub fn train(root: impl AsRef<Path>) -> Result<Stream<IdxExample>> {
    let stream = idx_dataset(root, IdxSplit::Train, SPEC)?;
    Ok(stream)
}

pub fn test(root: impl AsRef<Path>) -> Result<Stream<IdxExample>> {
    let stream = idx_dataset(root, IdxSplit::Test, SPEC)?;
    Ok(stream)
}
