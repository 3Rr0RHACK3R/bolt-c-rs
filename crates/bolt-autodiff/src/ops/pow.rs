use bolt_core::backend::{AddOp, CopyOp, DivOp, FillOp, LogOp, MulOp, PowOp, SubOp, SumOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;
use crate::ops::reduce_grad_to_shape;

pub struct PowBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl PowBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for PowBackward
where
    B: Backend
        + AddOp<D>
        + MulOp<D>
        + PowOp<D>
        + LogOp<D>
        + SubOp<D>
        + FillOp<D>
        + DivOp<D>
        + CopyOp<D>
        + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let base = ctx.saved(0);
        let exponent = ctx.saved(1);

        let one_val = D::one();
        let one = Tensor::full(&base.backend(), base.shape(), one_val)?;
        let exp_minus_one = exponent.sub(&one)?;
        let base_pow_exp_minus_one = base.pow(&exp_minus_one)?;
        let grad_lhs_full = grad_output.mul(&exponent)?.mul(&base_pow_exp_minus_one)?;

        let output = ctx.saved(2);
        let log_base = base.log()?;
        let grad_rhs_full = grad_output.mul(output)?.mul(&log_base)?;

        let grad_lhs = reduce_grad_to_shape(&grad_lhs_full, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(&grad_rhs_full, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }
}
