use bolt_core::backend::{CopyOp, ReshapeOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct ReshapeBackward {
    original_shape: Vec<usize>,
}

impl ReshapeBackward {
    pub fn new(original_shape: Vec<usize>) -> Self {
        Self { original_shape }
    }
}

impl<B, D> BackwardOp<B, D> for ReshapeBackward
where
    B: Backend + CopyOp<D> + ReshapeOp<D>,
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
        "ReshapeBackward"
    }
}
