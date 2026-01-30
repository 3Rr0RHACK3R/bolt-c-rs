use bolt_core::backend::{SqueezeOp, UnsqueezeOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct UnsqueezeBackward {
    axis: isize,
}

impl UnsqueezeBackward {
    pub(crate) fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl<B, D> BackwardOp<B, D> for UnsqueezeBackward
where
    B: Backend + SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        Ok(vec![Some(grad_output.squeeze_axis(self.axis)?)])
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}
