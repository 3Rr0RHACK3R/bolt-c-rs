mod binary;
mod reduce;
mod shape;

pub use binary::{AddBackward, MulBackward, SubBackward};
pub use reduce::{MeanBackward, SumBackward};
pub use shape::{ReshapeBackward, TransposeBackward};

use bolt_core::backend::{AddOp, CopyOp};
use bolt_core::{Backend, Tensor};

use crate::Float;
use crate::error::Result;

pub(crate) fn reduce_grad_to_shape<B, D>(
    grad: &Tensor<B, D>,
    target_shape: &[usize],
) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    let grad_shape = grad.shape();

    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    if target_shape.is_empty() {
        return sum_all(grad);
    }

    let grad_rank = grad_shape.len();
    let target_rank = target_shape.len();

    if grad_rank < target_rank {
        return Err(crate::error::Error::Core(bolt_core::Error::ShapeMismatch {
            lhs: grad_shape.to_vec(),
            rhs: target_shape.to_vec(),
        }));
    }

    let rank_diff = grad_rank - target_rank;
    let mut result = grad.clone();

    for i in 0..rank_diff {
        result = sum_axis(&result, 0)?;
        let _ = i;
    }

    for i in 0..target_rank {
        if target_shape[i] == 1 && result.shape()[i] != 1 {
            result = sum_axis(&result, i)?;
            let new_shape: Vec<usize> = result
                .shape()
                .iter()
                .enumerate()
                .map(|(j, &d)| if j == i { 1 } else { d })
                .collect();
            result = result.reshape(&new_shape)?;
        }
    }

    Ok(result)
}

fn sum_all<B, D>(tensor: &Tensor<B, D>) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    let mut result = tensor.clone();
    while result.rank() > 0 {
        result = sum_axis(&result, 0)?;
    }
    Ok(result)
}

pub(crate) fn sum_axis<B, D>(tensor: &Tensor<B, D>, axis: usize) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AddOp<D> + CopyOp<D>,
    D: Float,
{
    let shape = tensor.shape();
    if axis >= shape.len() {
        return Err(crate::error::Error::Core(bolt_core::Error::InvalidAxes(
            format!("axis {} out of bounds for rank {}", axis, shape.len()),
        )));
    }

    let axis_size = shape[axis];
    if axis_size == 0 {
        return Err(crate::error::Error::Core(bolt_core::Error::invalid_shape(
            "cannot reduce empty axis",
        )));
    }

    if axis_size == 1 {
        let new_shape: Vec<usize> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, &d)| d)
            .collect();
        let contiguous = tensor.contiguous()?;
        return Ok(contiguous.reshape(&new_shape)?);
    }

    let first_slice = tensor.slice(axis, 0, 1, 1)?;
    let reduced_shape: Vec<usize> = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, &d)| d)
        .collect();
    let first_contiguous = first_slice.contiguous()?;
    let mut result = first_contiguous.reshape(&reduced_shape)?;

    for i in 1..axis_size {
        let slice = tensor.slice(axis, i, i + 1, 1)?;
        let slice_contiguous = slice.contiguous()?;
        let slice = slice_contiguous.reshape(&reduced_shape)?;
        result = result.add(&slice)?;
    }

    Ok(result)
}

pub(crate) fn broadcast_to<B, D>(
    tensor: &Tensor<B, D>,
    target_shape: &[usize],
) -> Result<Tensor<B, D>>
where
    B: Backend<D> + CopyOp<D>,
    D: Float,
{
    if tensor.shape() == target_shape {
        return Ok(tensor.clone());
    }

    let src_shape = tensor.shape();
    let src_rank = src_shape.len();
    let target_rank = target_shape.len();

    let mut current = tensor.clone();

    if src_rank < target_rank {
        let mut new_shape = vec![1; target_rank - src_rank];
        new_shape.extend_from_slice(src_shape);
        current = current.reshape(&new_shape)?;
    }

    let layout = current
        .layout()
        .broadcast_to(&bolt_core::shape::ConcreteShape::from_slice(target_shape)?)?;

    let result = Tensor::from_parts(current.backend(), current.storage().clone(), layout);

    Ok(result.contiguous()?)
}
