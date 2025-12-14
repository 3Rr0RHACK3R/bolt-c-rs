use bolt_core::backend::{
    AbsOp, AddOp, BroadcastToOp, DivOp, FillOp, MulOp, ProdOp, ReshapeOp, SubOp, SumOp,
};
use bolt_core::{Backend, Float, Tensor};
use tinyvec::ArrayVec;

use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

fn zero_mask<B, D>(tensor: &Tensor<B, D>) -> Result<Tensor<B, D>>
where
    B: Backend + AbsOp<D> + AddOp<D> + FillOp<D> + SubOp<D> + DivOp<D>,
    D: Float,
{
    let abs = tensor.abs()?;
    let eps = Tensor::full(&tensor.backend(), abs.shape(), D::from_f64(1e-12))?;
    let denom = abs.add(&eps)?;
    let ratio = abs.div(&denom)?;
    let one = Tensor::full(&tensor.backend(), tensor.shape(), D::one())?;
    Ok(one.sub(&ratio)?)
}

fn eq_scalar<B, D>(tensor: &Tensor<B, D>, value: D) -> Result<Tensor<B, D>>
where
    B: Backend + AbsOp<D> + AddOp<D> + FillOp<D> + SubOp<D> + DivOp<D>,
    D: Float,
{
    let target = Tensor::full(&tensor.backend(), tensor.shape(), value)?;
    let diff = tensor.sub(&target)?;
    zero_mask(&diff)
}

pub struct ProdBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
}

impl ProdBackward {
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

impl<B, D> BackwardOp<B, D> for ProdBackward
where
    B: Backend
        + AddOp<D>
        + AbsOp<D>
        + BroadcastToOp<D>
        + DivOp<D>
        + FillOp<D>
        + MulOp<D>
        + ProdOp<D>
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
        let axes = self.axes_as_isize();

        let grad_bc = grad_output
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;
        let output_bc = output
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;

        let zeros_mask = zero_mask(input)?;
        let zero_count = match axes.as_ref() {
            Some(axes) => zeros_mask.sum(Some(axes), true)?,
            None => zeros_mask.sum(None, true)?,
        };

        let zero_eq_zero = eq_scalar(&zero_count, D::zero())?;
        let zero_eq_one = eq_scalar(&zero_count, D::one())?;

        let zero_eq_zero_bc = zero_eq_zero
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;
        let zero_eq_one_bc = zero_eq_one
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;

        let safe_input = input.add(&zeros_mask)?;
        let grad_no_zero = grad_bc.mul(&output_bc)?.div(&safe_input)?;
        let grad_no_zero = grad_no_zero.mul(&zero_eq_zero_bc)?;

        let input_no_zero = input.add(&zeros_mask)?;
        let non_zero_prod = match axes.as_ref() {
            Some(axes) => input_no_zero.prod(Some(axes), true)?,
            None => input_no_zero.prod(None, true)?,
        };
        let non_zero_prod_bc = non_zero_prod
            .reshape(&shape_with_ones)?
            .broadcast_to(&self.input_shape)?;

        let grad_single_zero = grad_bc.mul(&non_zero_prod_bc)?;
        let grad_single_zero = grad_single_zero.mul(&zeros_mask)?;
        let grad_single_zero = grad_single_zero.mul(&zero_eq_one_bc)?;

        let grad_input = grad_no_zero.add(&grad_single_zero)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}
