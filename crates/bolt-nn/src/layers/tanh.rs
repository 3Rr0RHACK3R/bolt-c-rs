use bolt_core::backend::{CopyOp, TanhOp};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

pub struct Tanh;

impl Default for Tanh {
    fn default() -> Self {
        Self
    }
}

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> Module<B, D> for Tanh
where
    B: BaseBackend + CopyOp<D> + TanhOp<D> + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        Ok(x.tanh()?)
    }
}
