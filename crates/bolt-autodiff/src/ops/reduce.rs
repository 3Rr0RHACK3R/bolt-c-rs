use bolt_core::backend::{AddOp, CopyOp, FillOp, MeanOp, MulOp, SumOp};
use bolt_core::shape;
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;
use crate::{Float, GradTensor};

pub struct SumBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
}

impl SumBackward {
    pub fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>) -> Self {
        Self { input_shape, axes }
    }
}

impl<B, D> BackwardOp<B, D> for SumBackward
where
    B: Backend<D> + AddOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let mut shape_with_ones = self.input_shape.clone();

        if let Some(ref axes) = self.axes {
            for &axis in axes {
                shape_with_ones[axis] = 1;
            }
        } else {
            shape_with_ones.fill(1);
        }

        let grad_reshaped = grad_output.reshape(&shape_with_ones)?;
        let grad_input = grad_reshaped.broadcast_to(&self.input_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

pub struct MeanBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
    count: usize,
}

impl MeanBackward {
    pub fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>, count: usize) -> Self {
        Self {
            input_shape,
            axes,
            count,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MeanBackward
where
    B: Backend<D> + AddOp<D> + FillOp<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let scale = D::one() / D::from_usize(self.count);
        let scaled =
            Tensor::full(&grad_output.backend(), grad_output.shape(), scale)?.mul(grad_output)?;

        let mut shape_with_ones = self.input_shape.clone();

        if let Some(ref axes) = self.axes {
            for &axis in axes {
                shape_with_ones[axis] = 1;
            }
        } else {
            shape_with_ones.fill(1);
        }

        let grad_reshaped = scaled.reshape(&shape_with_ones)?;
        let grad_input = grad_reshaped.broadcast_to(&self.input_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D> + AddOp<D> + FillOp<D> + SumOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn sum(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let input_shape = self_tensor.shape().to_vec();

        let result = self_tensor.sum(axes, keepdims)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().input(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let normalized_axes = axes.map(|a| bolt_core::shape::canonical_axes(a, input_shape.len())).transpose()?;
        let backward_op = SumBackward::new(input_shape, normalized_axes);

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

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D> + AddOp<D> + FillOp<D> + MulOp<D> + MeanOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn mean(&self, axes: Option<&[isize]>, keepdims: bool) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let input_shape = self_tensor.shape().to_vec();

        // Compute count for backward pass
        let count = match axes {
            None => self_tensor.numel(),
            Some(ax) => {
                let canonical = shape::canonical_axes(ax, input_shape.len())?;
                canonical.iter().map(|&a| input_shape[a]).product()
            }
        };

        // Clean delegation to Tensor API
        let result = self_tensor.mean(axes, keepdims)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().input(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let normalized_axes = axes.map(|a| bolt_core::shape::canonical_axes(a, input_shape.len())).transpose()?;
        let backward_op = MeanBackward::new(input_shape, normalized_axes, count);

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
