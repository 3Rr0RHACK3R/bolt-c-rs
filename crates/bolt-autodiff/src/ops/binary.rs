use bolt_core::{Backend, Tensor, backend::{AddOp, CopyOp, FillOp, MulOp, SubOp}};
use tinyvec::ArrayVec;

use crate::{
    Float, GradTensor,
    backward::{BackwardContext, BackwardOp, MAX_INPUTS},
    error::Result,
    ops::reduce_grad_to_shape,
};

pub struct AddBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl AddBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self { lhs_shape, rhs_shape }
    }
}

impl<B, D> BackwardOp<B, D> for AddBackward
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_lhs = reduce_grad_to_shape(grad_output, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(grad_output, &self.rhs_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

pub struct SubBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl SubBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self { lhs_shape, rhs_shape }
    }
}

impl<B, D> BackwardOp<B, D> for SubBackward
where
    B: Backend<D> + AddOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_lhs = reduce_grad_to_shape(grad_output, &self.lhs_shape)?;

        let neg_grad = Tensor::zeros(&grad_output.backend(), grad_output.shape())?
            .sub(grad_output)?;
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

pub struct MulBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl MulBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self { lhs_shape, rhs_shape }
    }
}

impl<B, D> BackwardOp<B, D> for MulBackward
where
    B: Backend<D> + AddOp<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let lhs = ctx.saved(0);
        let rhs = ctx.saved(1);

        let grad_lhs_full = grad_output.mul(rhs)?;
        let grad_rhs_full = grad_output.mul(lhs)?;

        let grad_lhs = reduce_grad_to_shape(&grad_lhs_full, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(&grad_rhs_full, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn add(&self, other: &GradTensor<'g, B, D>) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let other_tensor = other.tensor()?;

        let result = self_tensor.add(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = AddBackward::new(
            self_tensor.shape().to_vec(),
            other_tensor.shape().to_vec(),
        );

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other.handle());

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
    B: Backend<D> + AddOp<D> + SubOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn sub(&self, other: &GradTensor<'g, B, D>) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let other_tensor = other.tensor()?;

        let result = self_tensor.sub(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = SubBackward::new(
            self_tensor.shape().to_vec(),
            other_tensor.shape().to_vec(),
        );

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other.handle());

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
    B: Backend<D> + AddOp<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    pub fn mul(&self, other: &GradTensor<'g, B, D>) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let other_tensor = other.tensor()?;

        let result = self_tensor.mul(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().tensor(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![
            self_tensor.clone(),
            other_tensor.clone(),
        ]);
        let backward_op = MulBackward::new(
            self_tensor.shape().to_vec(),
            other_tensor.shape().to_vec(),
        );

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other.handle());

        Ok(self.graph().create_node(
            result,
            true,
            false,
            inputs,
            Some((Box::new(backward_op), saved_idx)),
        ))
    }
}
