use bolt_core::{Backend, Tensor, backend::{AddOp, CopyOp, FillOp, MulOp}};
use tinyvec::ArrayVec;

use crate::{
    Float, GradTensor,
    backward::{BackwardContext, BackwardOp, MAX_INPUTS},
    error::Result,
    ops::{broadcast_to, sum_axis},
};

pub struct SumBackward {
    input_shape: Vec<usize>,
}

impl SumBackward {
    pub fn new(input_shape: Vec<usize>, _axes: Option<Vec<usize>>) -> Self {
        Self { input_shape }
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
        let grad_input = broadcast_to(grad_output, &self.input_shape)?;
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
    count: usize,
}

impl MeanBackward {
    pub fn new(input_shape: Vec<usize>, _axes: Option<Vec<usize>>, count: usize) -> Self {
        Self { input_shape, count }
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
        let scaled = Tensor::full(&grad_output.backend(), grad_output.shape(), scale)?
            .mul(grad_output)?;
        let grad_input = broadcast_to(&scaled, &self.input_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

fn sum_impl<B, D>(tensor: &Tensor<B, D>, axes: Option<&[usize]>) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    match axes {
        None => {
            let mut result = tensor.clone();
            while result.rank() > 0 {
                result = sum_axis(&result, 0)?;
            }
            Ok(result)
        }
        Some(axes) => {
            let mut sorted_axes = axes.to_vec();
            sorted_axes.sort_unstable();
            sorted_axes.reverse();

            let mut result = tensor.clone();
            for &axis in &sorted_axes {
                result = sum_axis(&result, axis)?;
            }
            Ok(result)
        }
    }
}

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D> + AddOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn sum(&self, axes: Option<&[usize]>) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let input_shape = self_tensor.shape().to_vec();

        let result = sum_impl(&self_tensor, axes)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = SumBackward::new(input_shape, axes.map(|a| a.to_vec()));

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
    B: Backend<D> + AddOp<D> + FillOp<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn mean(&self, axes: Option<&[usize]>) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let input_shape = self_tensor.shape().to_vec();

        let count = match axes {
            None => self_tensor.numel(),
            Some(axes) => axes.iter().map(|&a| input_shape[a]).product(),
        };

        let sum_result = sum_impl(&self_tensor, axes)?;
        let scale = D::one() / D::from_usize(count);
        let result = Tensor::full(&sum_result.backend(), sum_result.shape(), scale)?
            .mul(&sum_result)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = MeanBackward::new(input_shape, axes.map(|a| a.to_vec()), count);

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
