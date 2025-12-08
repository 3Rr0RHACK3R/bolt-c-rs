use bolt_core::backend::{AddOp, CopyOp, FillOp, SubOp, SumOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;
use crate::ops::reduce_grad_to_shape;

pub struct SubBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl SubBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for SubBackward
where
    B: Backend + AddOp<D> + SubOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_lhs = reduce_grad_to_shape(grad_output, &self.lhs_shape)?;

        let neg_grad =
            Tensor::zeros(&grad_output.backend(), grad_output.shape())?.sub(grad_output)?;
        let grad_rhs = reduce_grad_to_shape(&neg_grad, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}
