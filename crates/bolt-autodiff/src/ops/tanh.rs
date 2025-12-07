use bolt_core::backend::{CopyOp, FillOp, MulOp, SubOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct TanhBackward;

impl TanhBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for TanhBackward
where
    B: Backend<D> + MulOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let output = ctx.saved(0);
        let output_squared = output.mul(output)?;
        let one = Tensor::full(&output.backend(), output.shape(), D::one())?;
        let one_minus_sqr = one.sub(&output_squared)?;
        let grad_input = grad_output.mul(&one_minus_sqr)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}
