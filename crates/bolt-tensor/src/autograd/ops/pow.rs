use bolt_core::Backend;
use bolt_core::backend::{CopyOp, FillOp, LogOp, MulOp, NegOp, PowOp, SubOp};
use bolt_core::dtype::Float;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct PowBackward;

impl PowBackward {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for PowBackward
where
    B: Backend
        + CopyOp<D>
        + FillOp<D>
        + LogOp<D>
        + MulOp<D>
        + NegOp<D>
        + PowOp<D>
        + SubOp<D>
        + 'static,
    D: Float + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let base = ctx.saved(0);
        let exp = ctx.saved(1);
        let out = ctx.saved(2);
        let backend = grad_output.backend();

        let one = Tensor::ones(&backend, base.shape())?;
        let exp_minus_one = exp.sub(&one)?;
        let base_pow_exp_minus_one = base.pow(&exp_minus_one)?;
        let grad_base = grad_output.mul(&exp)?.mul(&base_pow_exp_minus_one)?;

        let base_log = base.log()?;
        let grad_exp = grad_output.mul(out)?.mul(&base_log)?;

        Ok(vec![Some(grad_base), Some(grad_exp)])
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }
}
