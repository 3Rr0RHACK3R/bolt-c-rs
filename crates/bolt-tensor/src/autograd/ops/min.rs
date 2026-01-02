use bolt_core::Backend;
use bolt_core::backend::CopyOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp, utils};

pub(crate) struct MinBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
    keepdims: bool,
    output_shape: Vec<usize>,
}

impl MinBackward {
    pub(crate) fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>, keepdims: bool) -> Self {
        let output_shape = utils::sum_output_shape(&input_shape, axes.as_deref(), keepdims);
        Self {
            input_shape,
            axes,
            keepdims,
            output_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MinBackward
where
    B: Backend + CopyOp<D> + 'static,
    D: NativeType + PartialOrd + PartialEq + std::ops::Div<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let backend = grad_output.backend();
        let x = ctx.saved(0).to_vec()?;
        let gout = grad_output.to_vec()?;

        let input_numel: usize = self.input_shape.iter().product();
        let out_numel: usize = self.output_shape.iter().product();

        let mut mins: Vec<Option<D>> = vec![None; out_numel.max(1)];
        let mut counts = vec![0usize; out_numel.max(1)];

        let in_strides = utils::row_major_strides(&self.input_shape);
        let out_strides = utils::row_major_strides(&self.output_shape);

        for (flat, &v) in x.iter().enumerate() {
            let out_flat = utils::reduce_out_flat(
                flat,
                &self.input_shape,
                &in_strides,
                &out_strides,
                self.axes.as_deref(),
                self.keepdims,
            );
            match mins[out_flat] {
                None => {
                    mins[out_flat] = Some(v);
                    counts[out_flat] = 1;
                }
                Some(m) => {
                    if v < m {
                        mins[out_flat] = Some(v);
                        counts[out_flat] = 1;
                    } else if v == m {
                        counts[out_flat] += 1;
                    }
                }
            }
        }

        let mut grad_x = vec![D::default(); input_numel];
        for (flat, grad) in grad_x.iter_mut().enumerate() {
            let out_flat = utils::reduce_out_flat(
                flat,
                &self.input_shape,
                &in_strides,
                &out_strides,
                self.axes.as_deref(),
                self.keepdims,
            );
            if Some(x[flat]) == mins[out_flat] {
                let denom = D::from_usize(counts[out_flat].max(1));
                *grad = gout[out_flat] / denom;
            }
        }

        let grad_input = Tensor::from_vec(&backend, grad_x, &self.input_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "MinBackward"
    }
}
