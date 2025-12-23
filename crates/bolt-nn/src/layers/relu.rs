use bolt_core::backend::{CopyOp, ReluOp};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

pub struct Relu;

impl Relu {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> Module<B, D> for Relu
where
    B: BaseBackend + CopyOp<D> + ReluOp<D> + 'static,
    D: Float + PartialOrd + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        Ok(x.relu()?)
    }
}
