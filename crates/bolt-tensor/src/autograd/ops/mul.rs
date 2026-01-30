use bolt_core::backend::{CopyOp, MulOp, ReshapeOp, SumOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::utils;
use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct MulBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl MulBackward {
    pub(crate) fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MulBackward
where
    B: Backend + CopyOp<D> + MulOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    D: NativeType + std::ops::Mul<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let rhs = ctx.saved(1);
        let lhs = ctx.saved(0);
        let grad_lhs = utils::reduce_grad_to_shape(&grad_output.mul(rhs)?, &self.lhs_shape)?;
        let grad_rhs = utils::reduce_grad_to_shape(&grad_output.mul(lhs)?, &self.rhs_shape)?;
        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}
