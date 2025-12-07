use bolt_core::backend::{CopyOp, ReshapeOp, SumOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct ExpandBackward {
    input_shape: Vec<usize>,
    expanded_shape: Vec<usize>,
}

impl ExpandBackward {
    pub fn new(input_shape: Vec<usize>, expanded_shape: Vec<usize>) -> Self {
        Self {
            input_shape,
            expanded_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for ExpandBackward
where
    B: Backend<D> + CopyOp<D> + SumOp<D> + ReshapeOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let mut grad_tensor = grad_output.contiguous()?;
        if grad_tensor.shape() != self.expanded_shape.as_slice() {
            grad_tensor = grad_tensor.reshape(&self.expanded_shape)?;
        }
        let expanded_rank = self.expanded_shape.len();
        let input_rank = self.input_shape.len();
        let mut axes = Vec::new();
        let rank_gap = expanded_rank.saturating_sub(input_rank);
        for (idx, &expanded_dim) in self.expanded_shape.iter().enumerate() {
            let input_dim = if idx < rank_gap {
                1
            } else {
                self.input_shape[idx - rank_gap]
            };
            if input_dim == 1 && expanded_dim > 1 {
                axes.push(idx as isize);
            }
        }
        if !axes.is_empty() {
            grad_tensor = grad_tensor.sum(Some(&axes), true)?;
        }
        let grad_input = grad_tensor.reshape(&self.input_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}
