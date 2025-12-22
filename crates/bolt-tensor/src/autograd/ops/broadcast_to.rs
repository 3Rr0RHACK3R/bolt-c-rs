use bolt_core::Backend;
use bolt_core::backend::SumOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::{Error, Result};

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

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
    B: Backend + SumOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        if grad_output.shape() != self.output_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                lhs: grad_output.shape().to_vec(),
                rhs: self.output_shape.clone(),
            });
        }

        let mut axes = Vec::new();
        for (i, (&in_dim, &out_dim)) in self
            .input_shape
            .iter()
            .zip(self.output_shape.iter())
            .enumerate()
        {
            if in_dim == 1 && out_dim != 1 {
                axes.push(i as isize);
            }
        }

        let g = if axes.is_empty() {
            grad_output.clone()
        } else {
            let backend = grad_output.backend();
            let parts = backend.sum(
                grad_output.layout(),
                grad_output.storage(),
                Some(&axes),
                true,
            )?;
            Tensor::from_parts(backend.clone(), parts.storage, parts.layout)
        };

        Ok(vec![Some(g)])
    }

    fn name(&self) -> &'static str {
        "BroadcastToBackward"
    }
}
