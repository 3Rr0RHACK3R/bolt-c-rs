use bolt_core::Backend;
use bolt_core::backend::{ReshapeOp, SumOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::{Error, Result};

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};
use crate::autograd::utils;

pub(crate) struct BroadcastToBackward {
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
}

impl BroadcastToBackward {
    pub(crate) fn new(input_shape: Vec<usize>, output_shape: Vec<usize>) -> Self {
        Self {
            input_shape,
            output_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for BroadcastToBackward
where
    B: Backend + ReshapeOp<D> + SumOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        if grad_output.shape().as_slice() != self.output_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                lhs: grad_output.shape().to_vec(),
                rhs: self.output_shape.clone(),
            });
        }

        let g = utils::reduce_grad_to_shape(grad_output, &self.input_shape)?;
        Ok(vec![Some(g)])
    }

    fn name(&self) -> &'static str {
        "BroadcastToBackward"
    }
}
