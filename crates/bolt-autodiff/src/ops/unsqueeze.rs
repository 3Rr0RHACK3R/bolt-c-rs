use bolt_core::backend::{CopyOp, SqueezeOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct UnsqueezeBackward {
    axis: usize,
}

impl UnsqueezeBackward {
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }
}

impl<B, D> BackwardOp<B, D> for UnsqueezeBackward
where
    B: Backend<D> + CopyOp<D> + SqueezeOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_input = grad_output.squeeze_axis(self.axis as isize)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}
