use bolt_core::Backend;
use bolt_core::backend::{CopyOp, DivOp, MulOp, NegOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct DivBackward;

impl DivBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for DivBackward
where
    B: Backend + CopyOp<D> + DivOp<D> + MulOp<D> + NegOp<D> + 'static,
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
        let grad_lhs = grad_output.div(rhs)?;
        let rhs_squared = rhs.mul(rhs)?;
        let grad_rhs = grad_output.mul(lhs)?.neg()?.div(&rhs_squared)?;
        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}
