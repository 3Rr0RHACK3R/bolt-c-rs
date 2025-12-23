use bolt_core::Backend;
use bolt_core::backend::{CopyOp, DivOp, MulOp, NegOp, ReshapeOp, SumOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};
use crate::autograd::utils;

pub(crate) struct DivBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl DivBackward {
    pub(crate) fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self { lhs_shape, rhs_shape }
    }
}

impl<B, D> BackwardOp<B, D> for DivBackward
where
    B: Backend + CopyOp<D> + DivOp<D> + MulOp<D> + NegOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    D: NativeType
        + std::ops::Div<Output = D>
        + std::ops::Mul<Output = D>
        + std::ops::Neg<Output = D>
        + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let rhs = ctx.saved(1);
        let lhs = ctx.saved(0);
        let grad_lhs = utils::reduce_grad_to_shape(&grad_output.div(rhs)?, &self.lhs_shape)?;
        let rhs_squared = rhs.mul(rhs)?;
        let grad_rhs_unreduced = grad_output.mul(lhs)?.neg()?.div(&rhs_squared)?;
        let grad_rhs = utils::reduce_grad_to_shape(&grad_rhs_unreduced, &self.rhs_shape)?;
        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}
