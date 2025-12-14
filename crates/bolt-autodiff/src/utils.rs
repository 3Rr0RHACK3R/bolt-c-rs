use std::sync::Arc;

use bolt_core::backend::FillOp;
use bolt_core::error::{Error, Result};
use bolt_core::layout::Layout;
use bolt_core::{BaseBackend, Float, Tensor};

use crate::error::Result as AutodiffResult;
use crate::operations::Autodiff;

pub(crate) fn normalize_unsqueeze_axis(axis: isize, rank: usize) -> Result<usize> {
    let limit = rank + 1;
    let normalized_axis = if axis < 0 {
        let candidate = limit as isize + axis;
        if candidate < 0 {
            return Err(Error::InvalidAxes(format!(
                "unsqueeze axis {} out of bounds for rank {} (valid range: [{}, {}])",
                axis,
                rank,
                -(rank as isize + 1),
                rank
            )));
        }
        candidate as usize
    } else {
        axis as usize
    };

    if normalized_axis > rank {
        return Err(Error::InvalidAxes(format!(
            "unsqueeze axis {} out of bounds for rank {} (valid range: [{}, {}])",
            axis,
            rank,
            -(rank as isize + 1),
            rank
        )));
    }

    Ok(normalized_axis)
}

pub(crate) fn normalize_transpose_axis(axis: isize, rank: usize, name: &str) -> Result<usize> {
    let normalized = if axis < 0 {
        let candidate = rank as isize + axis;
        if candidate < 0 {
            return Err(Error::InvalidAxes(format!(
                "transpose {} axis {} out of bounds for rank {} (valid range: [{}, {}])",
                name,
                axis,
                rank,
                -(rank as isize),
                rank - 1
            )));
        }
        candidate as usize
    } else {
        axis as usize
    };

    if normalized >= rank {
        return Err(Error::InvalidAxes(format!(
            "transpose {} axis {} out of bounds for rank {} (valid range: [{}, {}])",
            name,
            axis,
            rank,
            -(rank as isize),
            rank - 1
        )));
    }

    Ok(normalized)
}

pub(crate) fn create_saved_tensor<B, D>(
    backend: &Arc<B>,
    storage: &B::Storage<D>,
    layout: &Layout,
) -> Tensor<B, D>
where
    B: BaseBackend,
    D: Float,
    B::Storage<D>: Clone,
{
    Tensor::from_parts(backend.clone(), storage.clone(), layout.clone())
}

pub(crate) fn create_backward_seed<B, D>(
    backend: &Arc<B>,
    loss_tensor: &Tensor<Autodiff<B, D>, D>,
) -> AutodiffResult<Tensor<B, D>>
where
    B: BaseBackend + FillOp<D>,
    D: Float,
{
    let seed = if loss_tensor.numel() == 1 {
        Tensor::full(backend, &[], D::one())?
    } else {
        Tensor::ones(backend, loss_tensor.shape())?
    };
    Ok(seed)
}
