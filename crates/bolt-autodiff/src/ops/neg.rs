use bolt_core::backend::{FillOp, SubOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct NegBackward;

impl NegBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for NegBackward
where
    B: Backend<D> + SubOp<D> + FillOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_input =
            Tensor::zeros(&grad_output.backend(), grad_output.shape())?.sub(grad_output)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}
