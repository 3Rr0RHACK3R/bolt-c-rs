use bolt_core::backend::{AbsOp, AddOp, CopyOp, DivOp, FillOp, MulOp};
use bolt_core::{Backend, Float, Tensor};
use tinyvec::ArrayVec;

use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct AbsBackward;

impl AbsBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for AbsBackward
where
    B: Backend + AbsOp<D> + AddOp<D> + DivOp<D> + MulOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let input = ctx.saved(0);

        let epsilon: D = D::from_f64(1e-8);
        let eps_tensor = Tensor::full(&input.backend(), input.shape(), epsilon)?;
        let abs_input = input.abs()?;
        let denom = abs_input.add(&eps_tensor)?;
        let sign = input.div(&denom)?;

        let grad_input = grad_output.mul(&sign)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "AbsBackward"
    }
}
