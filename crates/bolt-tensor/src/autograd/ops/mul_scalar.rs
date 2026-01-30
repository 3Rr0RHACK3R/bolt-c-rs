use bolt_core::backend::{CopyOp, MulScalarOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct MulScalarBackward<D> {
    scalar: D,
}

impl<D: NativeType> MulScalarBackward<D> {
    pub(crate) fn new(scalar: D) -> Self {
        Self { scalar }
    }
}

impl<B, D> BackwardOp<B, D> for MulScalarBackward<D>
where
    B: Backend + CopyOp<D> + MulScalarOp<D> + 'static,
    D: NativeType + std::ops::Mul<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        // d/dx (x * c) = c
        // So grad_input = grad_output * scalar
        let grad_input = grad_output.mul_scalar(self.scalar)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "MulScalarBackward"
    }
}
