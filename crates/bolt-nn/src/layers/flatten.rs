use bolt_core::backend::ReshapeOp;
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

pub struct Flatten;

impl Default for Flatten {
    fn default() -> Self {
        Self
    }
}

impl Flatten {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> Module<B, D> for Flatten
where
    B: BaseBackend + ReshapeOp<D> + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        let shape = x.shape();
        if shape.is_empty() {
            return Ok(x);
        }
        let batch = shape[0];
        let rest: usize = shape.as_slice()[1..].iter().product();
        Ok(x.reshape(&[batch, rest])?)
    }
}
