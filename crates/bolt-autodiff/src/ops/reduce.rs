use bolt_core::backend::{AddOp, CopyOp, FillOp, MulOp};
use bolt_core::{Backend, Tensor};
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct SumBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
}

impl SumBackward {
    pub fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>) -> Self {
        Self { input_shape, axes }
    }
}

impl<B, D> BackwardOp<B, D> for SumBackward
where
    B: Backend<D> + AddOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let mut shape_with_ones = self.input_shape.clone();

        if let Some(ref axes) = self.axes {
            for &axis in axes {
                shape_with_ones[axis] = 1;
            }
        } else {
            shape_with_ones.fill(1);
        }

        let grad_reshaped = grad_output.reshape(&shape_with_ones)?;
        let grad_input = grad_reshaped.broadcast_to(&self.input_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

pub struct MeanBackward {
    input_shape: Vec<usize>,
    axes: Option<Vec<usize>>,
    count: usize,
}

impl MeanBackward {
    pub fn new(input_shape: Vec<usize>, axes: Option<Vec<usize>>, count: usize) -> Self {
        Self {
            input_shape,
            axes,
            count,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MeanBackward
where
    B: Backend<D> + AddOp<D> + FillOp<D> + MulOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let scale = D::one() / D::from_usize(self.count);
        let scaled =
            Tensor::full(&grad_output.backend(), grad_output.shape(), scale)?.mul(grad_output)?;

        let mut shape_with_ones = self.input_shape.clone();

        if let Some(ref axes) = self.axes {
            for &axis in axes {
                shape_with_ones[axis] = 1;
            }
        } else {
            shape_with_ones.fill(1);
        }

        let grad_reshaped = scaled.reshape(&shape_with_ones)?;
        let grad_input = grad_reshaped.broadcast_to(&self.input_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}
