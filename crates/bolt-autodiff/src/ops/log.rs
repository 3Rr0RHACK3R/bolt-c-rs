use bolt_core::backend::{CopyOp, DivOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct LogBackward;

impl LogBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for LogBackward
where
    B: Backend<D> + DivOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let input = ctx.saved(0);
        let grad_input = grad_output.div(input)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}
