mod dataset;
mod header;
mod types;

pub use types::{IdxExample, IdxSpec, IdxSplit};

use std::path::Path;
use std::sync::Arc;

use crate::Result;
use crate::stream::Stream;

use dataset::{IdxDataset, IdxSource};

pub fn idx_dataset(
    root: impl AsRef<Path>,
    split: IdxSplit,
    spec: IdxSpec,
) -> Result<Stream<IdxExample>> {
    let root = root.as_ref();

    let (images, labels) = match split {
        IdxSplit::Train => (
            root.join("train-images-idx3-ubyte"),
            root.join("train-labels-idx1-ubyte"),
        ),
        IdxSplit::Test => (
            root.join("t10k-images-idx3-ubyte"),
            root.join("t10k-labels-idx1-ubyte"),
        ),
    };

    let dataset = Arc::new(IdxDataset::from_paths(spec, images, labels)?);
    let source = IdxSource::new(dataset)?;

    Ok(Stream::new(source))
}
