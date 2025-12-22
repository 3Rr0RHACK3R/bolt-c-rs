mod batch;
mod download;

use bolt_core::Backend;
use bolt_data::{IdxExample, IdxSpec, IdxSplit, Stream, idx_dataset};
use bolt_tensor::Tensor;
use std::path::Path;
use std::sync::Arc;

use crate::Result;

pub use batch::MnistBatch;
pub use download::ensure_downloaded;

pub const SPEC: IdxSpec = IdxSpec::new(28, 28, 1);
pub const NUM_CLASSES: usize = 10;
pub const INPUT_DIM: usize = 28 * 28;

pub struct MnistSample<B: Backend> {
    pub image: Tensor<B, f32>,
    pub label: i32,
}

pub fn train(root: impl AsRef<Path>) -> Result<Stream<IdxExample>> {
    let stream = idx_dataset(root, IdxSplit::Train, SPEC)?;
    Ok(stream)
}

pub fn test(root: impl AsRef<Path>) -> Result<Stream<IdxExample>> {
    let stream = idx_dataset(root, IdxSplit::Test, SPEC)?;
    Ok(stream)
}

pub fn to_tensor_label<B: Backend>(backend: &Arc<B>, ex: IdxExample) -> MnistSample<B> {
    let mut buf = Vec::with_capacity(ex.pixels.len());
    for p in ex.pixels {
        buf.push(p as f32);
    }
    let image = Tensor::from_vec(backend, buf, &[SPEC.channels, SPEC.rows, SPEC.cols])
        .expect("build tensor");
    MnistSample {
        image,
        label: ex.label as i32,
    }
}
