use bolt_core::Backend;
use bolt_core::backend::{CopyOp, NegOp, ReshapeOp, SumOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};
use crate::autograd::utils;

pub(crate) struct SubBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl SubBackward {
    pub(crate) fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self { lhs_shape, rhs_shape }
    }
}

impl<B, D> BackwardOp<B, D> for SubBackward
where
    B: Backend + CopyOp<D> + NegOp<D> + ReshapeOp<D> + SumOp<D> + 'static,
    D: NativeType + std::ops::Neg<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let grad_lhs = utils::reduce_grad_to_shape(grad_output, &self.lhs_shape)?;
        let grad_rhs = utils::reduce_grad_to_shape(grad_output, &self.rhs_shape)?.neg()?;
        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}
