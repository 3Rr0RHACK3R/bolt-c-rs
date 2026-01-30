use bolt_core::backend::CopyOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct AddScalarBackward;

impl AddScalarBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for AddScalarBackward
where
    B: Backend + CopyOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        // d/dx (x + c) = 1
        // So grad_input = grad_output * 1 = grad_output
        Ok(vec![Some(grad_output.clone())])
    }

    fn name(&self) -> &'static str {
        "AddScalarBackward"
    }
}
