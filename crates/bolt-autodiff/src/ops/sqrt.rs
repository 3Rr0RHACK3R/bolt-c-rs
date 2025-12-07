use bolt_core::backend::{CopyOp, DivOp, FillOp, MulOp};
use bolt_core::{Backend, Tensor};
use num_traits::cast;
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct SqrtBackward;

impl SqrtBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for SqrtBackward
where
    B: Backend<D> + DivOp<D> + MulOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let output = ctx.saved(0);
        let two: D = cast(2.0).unwrap();
        let two_tensor = Tensor::full(&output.backend(), output.shape(), two)?;
        let denom = two_tensor.mul(output)?;
        let grad_input = grad_output.div(&denom)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}
