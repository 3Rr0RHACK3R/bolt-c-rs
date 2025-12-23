use bolt_core::backend::ReshapeOp;
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{Module, Result};

pub struct Flatten;

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
    fn forward(&self, x: Tensor<B, D>, _train: bool) -> Result<Tensor<B, D>> {
        let shape = x.shape();
        if shape.is_empty() {
            return Ok(x);
        }
        let batch = shape[0];
        let rest: usize = shape[1..].iter().product();
        Ok(x.reshape(&[batch, rest])?)
    }
}

