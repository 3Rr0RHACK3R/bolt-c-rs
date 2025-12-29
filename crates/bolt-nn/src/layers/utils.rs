use bolt_core::backend::{BroadcastToOp, CopyOp, ReshapeOp, SqueezeOp, UnsqueezeOp};
use bolt_core::shape::normalize_axis;
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{Error, Result};

pub(crate) fn expand_to_input_shape<B, D>(
    param: &Tensor<B, D>,
    input_shape: &[usize],
    param_axes: &[isize],
) -> Result<Tensor<B, D>>
where
    B: BaseBackend
        + BroadcastToOp<D>
        + CopyOp<D>
        + ReshapeOp<D>
        + SqueezeOp<D>
        + UnsqueezeOp<D>
        + 'static,
    D: Float + 'static,
{
    let param_shape = param.shape();
    let input_rank = input_shape.len();
    let param_rank = param_shape.len();

    if param_rank != param_axes.len() {
        return Err(Error::Shape(format!(
            "expand_to_input_shape: param_axes.len() ({}) must equal param.rank() ({})",
            param_axes.len(),
            param_rank
        )));
    }

    if param_rank == input_rank {
        return Ok(param.clone());
    }

    let mut normalized_param_axes: Vec<usize> = Vec::with_capacity(param_axes.len());
    for &axis in param_axes {
        normalized_param_axes.push(normalize_axis(axis, input_rank)?);
    }

    let mut seen = vec![false; input_rank];
    for &axis in &normalized_param_axes {
        if seen[axis] {
            return Err(Error::Shape(format!(
                "expand_to_input_shape: duplicate axis {} in param_axes",
                axis
            )));
        }
        seen[axis] = true;
    }

    normalized_param_axes.sort_unstable();
    let mut param_axes_iter = normalized_param_axes.into_iter().peekable();

    let mut expanded = param.clone();

    for pos in 0..input_rank {
        if Some(&pos) == param_axes_iter.peek() {
            param_axes_iter.next();
        } else {
            expanded = expanded.unsqueeze(pos as isize)?;
        }
    }

    Ok(expanded)
}
