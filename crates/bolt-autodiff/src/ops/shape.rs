use bolt_core::{
    Backend, Tensor,
    backend::CopyOp,
};
use tinyvec::ArrayVec;

use crate::{
    Float,
    backward::{BackwardContext, BackwardOp, MAX_INPUTS},
    error::Result,
};

pub struct ReshapeBackward {
    original_shape: Vec<usize>,
}

impl ReshapeBackward {
    pub fn new(original_shape: Vec<usize>) -> Self {
        Self { original_shape }
    }
}

impl<B, D> BackwardOp<B, D> for ReshapeBackward
where
    B: Backend<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_contiguous = grad_output.contiguous()?;
        let grad_input = grad_contiguous.reshape(&self.original_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

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
    B: Backend<D> + CopyOp<D>,
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
