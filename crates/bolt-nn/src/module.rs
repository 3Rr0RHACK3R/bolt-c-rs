use bolt_core::BaseBackend;
use bolt_core::Float;
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Result};

pub trait Module<B, D>: Send + Sync
where
    B: BaseBackend,
    D: Float,
{
    fn forward(&self, x: Tensor<B, D>, ctx: &mut ForwardCtx) -> Result<Tensor<B, D>>;
}
