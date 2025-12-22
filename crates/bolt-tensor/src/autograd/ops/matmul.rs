use bolt_core::Backend;
use bolt_core::backend::{MatmulOp, TransposeOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct MatmulBackward;

impl MatmulBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for MatmulBackward
where
    B: Backend + MatmulOp<D> + TransposeOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let lhs = ctx.saved(0);
        let rhs = ctx.saved(1);

        let rhs_t = rhs.transpose(0, 1)?;
        let lhs_t = lhs.transpose(0, 1)?;

        let grad_lhs = grad_output.matmul(&rhs_t)?;
        let grad_rhs = lhs_t.matmul(grad_output)?;

        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}
