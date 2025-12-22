use bolt_core::Backend;
use bolt_core::backend::{CopyOp, NegOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct NegBackward;

impl NegBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for NegBackward
where
    B: Backend + CopyOp<D> + NegOp<D> + 'static,
    D: NativeType + std::ops::Neg<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let grad_input = grad_output.neg()?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}
