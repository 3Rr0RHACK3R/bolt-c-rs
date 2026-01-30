use bolt_core::backend::CopyOp;
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;
use bolt_core::Backend;

use crate::autograd::{utils, BackwardContext, BackwardOp};
use crate::Tensor;

pub(crate) struct ProdBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
    keepdims: bool,
    output_shape: Vec<usize>,
}

impl ProdBackward {
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

impl<B, D> BackwardOp<B, D> for ProdBackward
where
    B: Backend + CopyOp<D> + 'static,
    D: NativeType + PartialEq + std::ops::Mul<Output = D> + std::ops::Div<Output = D> + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let backend = grad_output.backend();
        let x = ctx.saved(0).to_vec()?;
        let gout = grad_output.to_vec()?;
        let zero = D::default();

        let input_numel: usize = self.input_shape.iter().product();
        let out_numel: usize = self.output_shape.iter().product();

        let mut prod_nonzero = vec![D::one(); out_numel.max(1)];
        let mut zero_count = vec![0usize; out_numel.max(1)];

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
            if v == zero {
                zero_count[out_flat] += 1;
            } else {
                prod_nonzero[out_flat] = prod_nonzero[out_flat] * v;
            }
        }

        let mut grad_x = vec![zero; input_numel];
        for (flat, grad) in grad_x.iter_mut().enumerate() {
            let out_flat = utils::reduce_out_flat(
                flat,
                &self.input_shape,
                &in_strides,
                &out_strides,
                self.axes.as_deref(),
                self.keepdims,
            );
            let g = gout[out_flat];
            match zero_count[out_flat] {
                0 => {
                    *grad = g * prod_nonzero[out_flat] / x[flat];
                }
                1 => {
                    if x[flat] == zero {
                        *grad = g * prod_nonzero[out_flat];
                    }
                }
                _ => {}
            }
        }

        let grad_input = Tensor::from_vec(&backend, grad_x, &self.input_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}
