use bolt_core::backend::{CopyOp, MulOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct ExpBackward;

impl ExpBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for ExpBackward
where
    B: Backend<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let output = ctx.saved(0);
        let grad_input = grad_output.mul(output)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}
