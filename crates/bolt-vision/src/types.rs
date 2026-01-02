use bolt_core::{backend::CopyOp, Backend, NativeType};
use bolt_tensor::{Tensor, ToBackend};

pub enum ImageLayout {
    NHWC,
    NCHW,
}

pub struct ImageAndLabel<B: Backend, D: NativeType> {
    pub image: Tensor<B, D>,
    pub label: i32,
}

impl<B1: Backend + CopyOp<D>, B2: Backend, D: NativeType> ToBackend<B2> for ImageAndLabel<B1, D> {
    type Output = ImageAndLabel<B2, D>;
    fn to_backend(self, backend: &std::sync::Arc<B2>) -> bolt_core::Result<Self::Output> {
        Ok(ImageAndLabel {
            image: self.image.to_backend(backend)?,
            label: self.label,
        })
    }
}
