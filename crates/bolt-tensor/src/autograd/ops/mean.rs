use bolt_core::Backend;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::Float;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp, utils};

pub(crate) struct MeanBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
    keepdims: bool,
    reduce_count: usize,
}

impl MeanBackward {
    pub(crate) fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>, keepdims: bool) -> Self {
        let reduce_count = match axes.as_deref() {
            None => input_shape.iter().product(),
            Some([]) => 1,
            Some(axes) => axes.iter().map(|&a| input_shape[a]).product(),
        };

        Self {
            input_shape,
            axes,
            keepdims,
            reduce_count: reduce_count.max(1),
        }
    }
}

impl<B, D> BackwardOp<B, D> for MeanBackward
where
    B: Backend + CopyOp<D> + 'static,
    D: Float + 'static,
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

        let mut expanded = utils::expand_sum_grad(
            &backend,
            &gout,
            &out_shape,
            &self.input_shape,
            self.axes.as_deref(),
            self.keepdims,
        )?
        .to_vec()?;

        let scale = D::from_f64(1.0 / (self.reduce_count as f64));
        for x in &mut expanded {
            *x = *x * scale;
        }
        let grad_input = Tensor::from_vec(&backend, expanded, &self.input_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}
