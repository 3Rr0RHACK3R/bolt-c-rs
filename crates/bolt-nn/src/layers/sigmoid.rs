use bolt_core::backend::{CopyOp, SigmoidOp};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> Module<B, D> for Sigmoid
where
    B: BaseBackend + CopyOp<D> + SigmoidOp<D> + 'static,
    D: Float + std::ops::Mul<Output = D> + std::ops::Sub<Output = D> + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        Ok(x.sigmoid()?)
    }
}
