use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct ConcatBackward {
    axis: usize,
    input_shapes: Vec<Vec<usize>>,
}

impl ConcatBackward {
    pub(crate) fn new(axis: usize, input_shapes: Vec<Vec<usize>>) -> Self {
        Self { axis, input_shapes }
    }
}

impl<B, D> BackwardOp<B, D> for ConcatBackward
where
    B: Backend + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let mut grads = Vec::with_capacity(self.input_shapes.len());
        let mut start = 0;

        for input_shape in &self.input_shapes {
            let axis_size = input_shape[self.axis];
            let end = start + axis_size;

            // Slice the gradient along the concatenation axis
            let grad_slice = grad_output.slice(self.axis, start, end, 1)?;
            grads.push(Some(grad_slice));

            start = end;
        }

        Ok(grads)
    }

    fn name(&self) -> &'static str {
        "ConcatBackward"
    }
}
