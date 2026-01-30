use bolt_core::backend::TransposeOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct TransposeBackward {
    axis_a: isize,
    axis_b: isize,
}

impl TransposeBackward {
    pub(crate) fn new(axis_a: isize, axis_b: isize) -> Self {
        Self { axis_a, axis_b }
    }
}

impl<B, D> BackwardOp<B, D> for TransposeBackward
where
    B: Backend + TransposeOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        Ok(vec![Some(grad_output.transpose(self.axis_a, self.axis_b)?)])
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}
