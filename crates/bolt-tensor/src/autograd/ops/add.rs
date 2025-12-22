use bolt_core::Backend;
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct AddBackward;

impl AddBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for AddBackward
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        Ok(vec![Some(grad_output.clone()), Some(grad_output.clone())])
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}
