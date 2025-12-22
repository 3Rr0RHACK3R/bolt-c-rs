use bolt_core::Backend;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::{Error, Result};

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp, utils};

pub(crate) struct SumBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
    keepdims: bool,
}

impl SumBackward {
    pub(crate) fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>, keepdims: bool) -> Self {
        Self {
            input_shape,
            axes,
            keepdims,
        }
    }
}

impl<B, D> BackwardOp<B, D> for SumBackward
where
    B: Backend + CopyOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let backend = grad_output.backend();
        let gout = grad_output.to_vec()?;

        let out_shape =
            utils::sum_output_shape(&self.input_shape, self.axes.as_deref(), self.keepdims);
        if out_shape.iter().product::<usize>() != gout.len() {
            return Err(Error::ShapeMismatch {
                lhs: grad_output.shape().to_vec(),
                rhs: out_shape,
            });
        }

        let grad_input = utils::expand_sum_grad(
            &backend,
            &gout,
            &out_shape,
            &self.input_shape,
            self.axes.as_deref(),
            self.keepdims,
        )?;

        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}
