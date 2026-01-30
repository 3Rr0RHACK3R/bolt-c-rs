use bolt_core::backend::{CopyOp, DivScalarOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct DivScalarBackward<D> {
    scalar: D,
}

impl<D: NativeType> DivScalarBackward<D> {
    pub(crate) fn new(scalar: D) -> Self {
        Self { scalar }
    }
}

impl<B, D> BackwardOp<B, D> for DivScalarBackward<D>
where
    B: Backend + CopyOp<D> + DivScalarOp<D> + 'static,
    D: NativeType + std::ops::Div<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        // d/dx (x / c) = 1/c
        // So grad_input = grad_output / scalar
        let grad_input = grad_output.div_scalar(self.scalar)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "DivScalarBackward"
    }
}
