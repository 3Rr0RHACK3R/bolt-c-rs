use bolt_core::Backend;
use bolt_core::backend::{ReshapeOp, SumOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};
use crate::autograd::utils;

pub(crate) struct AddBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl AddBackward {
    pub(crate) fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self { lhs_shape, rhs_shape }
    }
}

impl<B, D> BackwardOp<B, D> for AddBackward
where
    B: Backend + ReshapeOp<D> + SumOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let grad_lhs = utils::reduce_grad_to_shape(grad_output, &self.lhs_shape)?;
        let grad_rhs = utils::reduce_grad_to_shape(grad_output, &self.rhs_shape)?;
        Ok(vec![Some(grad_lhs), Some(grad_rhs)])
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}
