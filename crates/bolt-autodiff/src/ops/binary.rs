use bolt_core::{
    Backend, Tensor,
    backend::{AddOp, CopyOp, FillOp, MatmulOp, MulOp, SubOp, SumOp},
};
use tinyvec::ArrayVec;

use crate::{
    Float,
    backward::{BackwardContext, BackwardOp, MAX_INPUTS},
    error::Result,
    ops::reduce_grad_to_shape,
};

pub struct AddBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl AddBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for AddBackward
where
    B: Backend<D> + AddOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_lhs = reduce_grad_to_shape(grad_output, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(grad_output, &self.rhs_shape)?;
        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

pub struct SubBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl SubBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for SubBackward
where
    B: Backend<D> + AddOp<D> + SubOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let grad_lhs = reduce_grad_to_shape(grad_output, &self.lhs_shape)?;

        let neg_grad =
            Tensor::zeros(&grad_output.backend(), grad_output.shape())?.sub(grad_output)?;
        let grad_rhs = reduce_grad_to_shape(&neg_grad, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

pub struct MulBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

pub struct MatmulBackward {
    lhs_shape: Vec<usize>,
    rhs_shape: Vec<usize>,
}

impl MulBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MulBackward
where
    B: Backend<D> + AddOp<D> + MulOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let lhs = ctx.saved(0);
        let rhs = ctx.saved(1);

        let grad_lhs_full = grad_output.mul(rhs)?;
        let grad_rhs_full = grad_output.mul(lhs)?;

        let grad_lhs = reduce_grad_to_shape(&grad_lhs_full, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(&grad_rhs_full, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

impl MatmulBackward {
    pub fn new(lhs_shape: Vec<usize>, rhs_shape: Vec<usize>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
        }
    }
}

impl<B, D> BackwardOp<B, D> for MatmulBackward
where
    B: Backend<D> + AddOp<D> + MatmulOp<D> + CopyOp<D> + SumOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let lhs = ctx.saved(0);
        let rhs = ctx.saved(1);

        let grad_lhs = grad_output.matmul(&rhs.transpose(1, 0)?)?;
        let grad_rhs = lhs.transpose(1, 0)?.matmul(grad_output)?;

        let grad_lhs = reduce_grad_to_shape(&grad_lhs, &self.lhs_shape)?;
        let grad_rhs = reduce_grad_to_shape(&grad_rhs, &self.rhs_shape)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_lhs));
        result.push(Some(grad_rhs));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}
