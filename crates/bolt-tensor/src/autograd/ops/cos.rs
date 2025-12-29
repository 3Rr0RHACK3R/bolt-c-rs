use bolt_core::Backend;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::Float;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct CosBackward;

impl CosBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for CosBackward
where
    B: Backend + CopyOp<D> + 'static,
    D: Float + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let backend = grad_output.backend();
        let gout = grad_output.to_vec()?;
        let x = ctx.saved(0).to_vec()?;
        let mut grad = Vec::with_capacity(gout.len());
        for (g, v) in gout.into_iter().zip(x.into_iter()) {
            grad.push(-g * v.sin());
        }
        let grad_input = Tensor::from_vec(&backend, grad, ctx.saved(0).shape().as_slice())?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "CosBackward"
    }
}
