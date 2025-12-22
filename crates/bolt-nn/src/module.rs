use bolt_core::BaseBackend;
use bolt_core::Float;
use bolt_tensor::Tensor;

use crate::Result;

pub trait Module<B, D>: Send + Sync
where
    B: BaseBackend,
    D: Float,
{
    fn forward(&self, x: Tensor<B, D>, train: bool) -> Result<Tensor<B, D>>;
}
