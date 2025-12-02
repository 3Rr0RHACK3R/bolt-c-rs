use bolt_core::{
    backend::{AddOp, CopyOp, FillOp, MatmulOp, MeanOp, MulOp, SubOp, SumOp},
    shape, Backend, Tensor,
};
use tinyvec::ArrayVec;

use crate::error::Result;
use crate::graph::MAX_SHAPE_RANK;
use crate::ops::{
    AddBackward, MatmulBackward, MeanBackward, MulBackward, ReshapeBackward, SubBackward,
    SumBackward, TransposeBackward,
};
use crate::{Float, Graph, Handle};

#[derive(Clone, Copy)]
pub struct GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    graph: &'g Graph<B, D>,
    handle: Handle,
}

impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub(crate) fn new(graph: &'g Graph<B, D>, handle: Handle) -> Self {
        Self { graph, handle }
    }

    pub fn tensor(&self) -> Result<Tensor<B, D>> {
        self.graph.get_tensor(self.handle)
    }

    pub fn handle(&self) -> Handle {
        self.handle
    }

    pub fn shape(&self) -> Result<ArrayVec<[usize; MAX_SHAPE_RANK]>> {
        self.graph.get_node_shape(self.handle)
    }

    pub fn requires_grad(&self) -> Result<bool> {
        self.graph.get_node_requires_grad(self.handle)
    }

    pub fn is_leaf(&self) -> Result<bool> {
        self.graph.get_node_is_leaf(self.handle)
    }

    pub fn as_constant(&self) -> Result<GradTensor<'g, B, D>> {
        let tensor = self.tensor()?;
        Ok(self.graph.constant(&tensor))
    }

    pub fn set_requires_grad(&self, requires_grad: bool) -> Result<()> {
        self.graph
            .set_node_requires_grad(self.handle, requires_grad)
    }

    pub fn graph(&self) -> &'g Graph<B, D> {
        self.graph
    }
}

pub trait TensorLike<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, graph: &'g Graph<B, D>) -> GradTensor<'g, B, D>;
}

impl<'g, B, D> TensorLike<'g, B, D> for GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, _graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        self
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for &GradTensor<'g, B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, _graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        self.clone()
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for Tensor<B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        graph.constant(&self)
    }
}

impl<'g, B, D> TensorLike<'g, B, D> for &Tensor<B, D>
where
    B: Backend<D>,
    D: Float,
{
    fn into_node(self, graph: &'g Graph<B, D>) -> GradTensor<'g, B, D> {
        graph.constant(self)
    }
}

// Binary operations
impl<'g, B, D> GradTensor<'g, B, D>
where
    B: Backend<D> + AddOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    pub fn add<T>(&self, other: T) -> Result<GradTensor<'g, B, D>>
    where
        T: TensorLike<'g, B, D>,
    {
        let self_tensor = self.tensor()?;
        let other_node = other.into_node(self.graph());
        let other_tensor = other_node.tensor()?;

        let result = self_tensor.add(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other_node.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().constant(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op =
            AddBackward::new(self_tensor.shape().to_vec(), other_tensor.shape().to_vec());

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other_node.handle());

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
    B: Backend<D> + AddOp<D> + SubOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    pub fn sub<T>(&self, other: T) -> Result<GradTensor<'g, B, D>>
    where
        T: TensorLike<'g, B, D>,
    {
        let self_tensor = self.tensor()?;
        let other_node = other.into_node(self.graph());
        let other_tensor = other_node.tensor()?;

        let result = self_tensor.sub(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other_node.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().constant(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op =
            SubBackward::new(self_tensor.shape().to_vec(), other_tensor.shape().to_vec());

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other_node.handle());

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
    B: Backend<D> + AddOp<D> + MulOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    pub fn mul<T>(&self, other: T) -> Result<GradTensor<'g, B, D>>
    where
        T: TensorLike<'g, B, D>,
    {
        let self_tensor = self.tensor()?;
        let other_node = other.into_node(self.graph());
        let other_tensor = other_node.tensor()?;

        let result = self_tensor.mul(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other_node.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().constant(&result));
        }

        let saved_idx = self
            .graph()
            .save_tensors_for_backward(vec![self_tensor.clone(), other_tensor.clone()]);
        let backward_op =
            MulBackward::new(self_tensor.shape().to_vec(), other_tensor.shape().to_vec());

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other_node.handle());

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
    B: Backend<D> + AddOp<D> + MatmulOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    pub fn matmul<T>(&self, other: T) -> Result<GradTensor<'g, B, D>>
    where
        T: TensorLike<'g, B, D>,
    {
        let self_tensor = self.tensor()?;
        let other_node = other.into_node(self.graph());
        let other_tensor = other_node.tensor()?;

        let result = self_tensor.matmul(&other_tensor)?;

        let self_requires_grad = self.requires_grad()?;
        let other_requires_grad = other_node.requires_grad()?;
        let requires_grad =
            (self_requires_grad || other_requires_grad) && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().constant(&result));
        }

        let saved_idx = self
            .graph()
            .save_tensors_for_backward(vec![self_tensor.clone(), other_tensor.clone()]);
        let backward_op =
            MatmulBackward::new(self_tensor.shape().to_vec(), other_tensor.shape().to_vec());

        let mut inputs = ArrayVec::new();
        inputs.push(self.handle());
        inputs.push(other_node.handle());

        Ok(self.graph().create_node(
            result,
            true,
            false,
            inputs,
            Some((Box::new(backward_op), saved_idx)),
        ))
    }
}

// Reduction operations
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
            return Ok(self.graph().constant(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let normalized_axes = axes
            .map(|a| shape::canonical_axes(a, input_shape.len()))
            .transpose()?;
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

        let count = match axes {
            None => self_tensor.numel(),
            Some(ax) => {
                let canonical = shape::canonical_axes(ax, input_shape.len())?;
                canonical.iter().map(|&a| input_shape[a]).product()
            }
        };

        let result = self_tensor.mean(axes, keepdims)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().constant(&result));
        }

        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let normalized_axes = axes
            .map(|a| shape::canonical_axes(a, input_shape.len()))
            .transpose()?;
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

// Shape operations
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
            return Ok(self.graph().constant(&result));
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

    pub fn transpose(&self, axis_a: isize, axis_b: isize) -> Result<GradTensor<'g, B, D>> {
        let self_tensor = self.tensor()?;
        let rank = self_tensor.shape().len();

        let result = self_tensor.transpose(axis_a, axis_b)?;

        let requires_grad = self.requires_grad()? && self.graph().is_grad_enabled();

        if !requires_grad {
            return Ok(self.graph().constant(&result));
        }

        let norm_a = shape::normalize_axis(axis_a, rank)?;
        let norm_b = shape::normalize_axis(axis_b, rank)?;
        let saved_idx = self.graph().save_tensors_for_backward(vec![]);
        let backward_op = TransposeBackward::new(norm_a, norm_b);

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
