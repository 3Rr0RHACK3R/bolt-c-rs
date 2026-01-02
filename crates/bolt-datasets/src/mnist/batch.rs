use bolt_core::backend::{Backend, CopyOp};
use bolt_tensor::Tensor;

pub struct MnistBatch<B: Backend> {
    pub images: Tensor<B, f32>,
    pub labels: Tensor<B, i32>,
}

impl<B1: Backend + CopyOp<f32> + CopyOp<i32>, B2: Backend> bolt_tensor::ToBackend<B2> for MnistBatch<B1> {
    type Output = MnistBatch<B2>;
    fn to_backend(self, backend: &std::sync::Arc<B2>) -> bolt_core::Result<Self::Output> {
        Ok(MnistBatch {
            images: self.images.to_backend(backend)?,
            labels: self.labels.to_backend(backend)?,
        })
    }
}
