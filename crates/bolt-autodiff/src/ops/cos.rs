use bolt_core::backend::{CopyOp, FillOp, MulOp, SinOp, SubOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct CosBackward;

impl CosBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for CosBackward
where
    B: Backend + SinOp<D> + MulOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let input = ctx.saved(0);
        let sin_input = input.sin()?;
        let neg_sin = Tensor::zeros(&sin_input.backend(), sin_input.shape())?.sub(&sin_input)?;
        let grad_input = grad_output.mul(&neg_sin)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "CosBackward"
    }
}
