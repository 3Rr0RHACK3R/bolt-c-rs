use bolt_core::Backend;
use bolt_core::backend::{CopyOp, MulOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct MulBackward;

impl MulBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for MulBackward
where
    B: Backend + CopyOp<D> + MulOp<D> + 'static,
    D: NativeType + std::ops::Mul<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let rhs = ctx.saved(1);
        let lhs = ctx.saved(0);
        let grad_lhs = grad_output.mul(rhs)?;
        let grad_rhs = grad_output.mul(lhs)?;
        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}
