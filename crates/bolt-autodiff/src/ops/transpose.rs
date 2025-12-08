use bolt_core::backend::{CopyOp, TransposeOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct TransposeBackward {
    axis_a: usize,
    axis_b: usize,
}

impl TransposeBackward {
    pub fn new(axis_a: usize, axis_b: usize) -> Self {
        Self { axis_a, axis_b }
    }
}

impl<B, D> BackwardOp<B, D> for TransposeBackward
where
    B: Backend + CopyOp<D> + TransposeOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_transposed = grad_output.transpose(self.axis_a as isize, self.axis_b as isize)?;
        let grad_input = grad_transposed.contiguous()?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}
