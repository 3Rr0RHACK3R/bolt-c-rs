use bolt_core::{Backend, Tensor, backend::{AddOp, CopyOp}};
use tinyvec::ArrayVec;

use crate::{
    Float, GradTensor,
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
        let grad_transposed = grad_output.transpose(self.axis_a, self.axis_b)?;
        let grad_input = grad_transposed.contiguous()?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn reshape(&self, shape: &[usize]) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let original_shape = self_tensor.shape().to_vec();

        let result = self_tensor.reshape(shape)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = ReshapeBackward::new(original_shape);

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());

        Ok(self.graph().create_node(
            result,
            true,
            false,
            inputs,
            Some((Box::new(backward_op), saved_idx)),
        ))
    }

    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;

        let result = self_tensor.transpose(axis_a, axis_b)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = TransposeBackward::new(axis_a, axis_b);

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());

        Ok(self.graph().create_node(
            result,
            true,
            false,
            inputs,
            Some((Box::new(backward_op), saved_idx)),
        ))
    }
}
