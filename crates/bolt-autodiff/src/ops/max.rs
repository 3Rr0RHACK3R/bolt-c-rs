use bolt_core::backend::{AbsOp, AddOp, BroadcastToOp, DivOp, FillOp, MulOp, ReshapeOp, SubOp, SumOp};
use bolt_core::{Backend, Tensor};
use num_traits::cast;
use tinyvec::ArrayVec;

use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;
use crate::Float;

fn zero_mask<B, D>(tensor: &Tensor<B, D>) -> Result<Tensor<B, D>>
where
    B: Backend<D> + AbsOp<D> + AddOp<D> + FillOp<D> + SubOp<D> + DivOp<D>,
    D: Float,
{
    let abs = tensor.abs()?;
    let eps = Tensor::full(&tensor.backend(), abs.shape(), cast::<_, D>(1e-12).unwrap())?;
    let denom = abs.add(&eps)?;
    let ratio = abs.div(&denom)?;
    let one = Tensor::full(&tensor.backend(), tensor.shape(), D::one())?;
    Ok(one.sub(&ratio)?)
}

pub struct MaxBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
}

impl MaxBackward {
    pub fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>) -> Self {
        Self { input_shape, axes }
    }

    fn shape_with_ones(&self) -> Vec<usize> {
        let mut shape = self.input_shape.clone();
        if let Some(axes) = &self.axes {
            for &axis in axes {
                shape[axis] = 1;
            }
        } else {
            shape.fill(1);
        }
        shape
    }

    fn axes_as_isize(&self) -> Option<Vec<isize>> {
        self.axes
            .as_ref()
            .map(|axes| axes.iter().map(|&a| a as isize).collect())
    }
}

impl<B, D> BackwardOp<B, D> for MaxBackward
where
    B: Backend<D>
        + AddOp<D>
        + AbsOp<D>
        + BroadcastToOp<D>
        + DivOp<D>
        + FillOp<D>
        + MulOp<D>
        + ReshapeOp<D>
        + SubOp<D>
        + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let input = ctx.saved(0);
        let output = ctx.saved(1);

        let shape_with_ones = self.shape_with_ones();

        let output_bc = output
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;
        let diff = output_bc.sub(input)?;
        let mask = zero_mask(&diff)?;

        let axes = self.axes_as_isize();
        let count = match axes {
            Some(ref axes) => mask.sum(Some(axes), true)?,
            None => mask.sum(None, true)?,
        };
        let count_bc = count
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;

        let grad_bc = grad_output
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;
        let grad_per_max = grad_bc.div(&count_bc)?;
        let grad_input = grad_per_max.mul(&mask)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}
