use bolt_core::backend::{CopyOp, ReshapeOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct SqueezeBackward {
    original_shape: Vec<usize>,
}

impl SqueezeBackward {
    pub fn new(original_shape: Vec<usize>) -> Self {
        Self { original_shape }
    }
}

impl<B, D> BackwardOp<B, D> for SqueezeBackward
where
    B: Backend<D> + CopyOp<D> + ReshapeOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_contiguous = grad_output.contiguous()?;
        let grad_input = grad_contiguous.reshape(&self.original_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }
}
