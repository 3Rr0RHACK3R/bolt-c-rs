use bolt_core::backend::ReshapeOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct ReshapeBackward {
    input_shape: Vec<usize>,
}

impl ReshapeBackward {
    pub(crate) fn new(input_shape: Vec<usize>) -> Self {
        Self { input_shape }
    }
}

impl<B, D> BackwardOp<B, D> for ReshapeBackward
where
    B: Backend + ReshapeOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        Ok(vec![Some(grad_output.reshape(&self.input_shape)?)])
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}
