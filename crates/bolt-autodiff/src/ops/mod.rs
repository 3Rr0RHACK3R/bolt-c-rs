mod binary;
mod reduce;
mod shape;

pub use binary::{AddBackward, MulBackward, SubBackward};
pub use reduce::{MeanBackward, SumBackward};
pub use shape::{ReshapeBackward, TransposeBackward};

use bolt_core::backend::{AddOp, SumOp};
use bolt_core::{Backend, Tensor};

use crate::Float;
use crate::error::Result;

pub(crate) fn reduce_grad_to_shape<B, D>(
    grad: &Tensor<B, D>,
    target_shape: &[usize],
) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AddOp<D> + SumOp<D>,
    D: Float,
{
    let grad_shape = grad.shape();

    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    if target_shape.is_empty() {
        return Ok(grad.sum(None, false)?);
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
        result = result.sum(Some(&[0]), false)?;
        let _ = i;
    }

    for i in 0..target_rank {
        if target_shape[i] == 1 && result.shape()[i] != 1 {
            result = result.sum(Some(&[i as isize]), true)?;
        }
    }

    Ok(result)
}
