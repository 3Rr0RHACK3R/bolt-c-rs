use bolt_core::backend::{AddOp, CopyOp, MulOp, SumOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;
use crate::ops::reduce_grad_to_shape;

pub struct MulBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl MulBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MulBackward
where
    B: Backend<D> + AddOp<D> + MulOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let lhs = ctx.saved(0);
        let rhs = ctx.saved(1);

        let grad_lhs_full = grad_output.mul(rhs)?;
        let grad_rhs_full = grad_output.mul(lhs)?;

        let grad_lhs = reduce_grad_to_shape(&grad_lhs_full, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(&grad_rhs_full, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}
