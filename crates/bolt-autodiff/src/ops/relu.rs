use bolt_core::backend::{AbsOp, AddOp, CopyOp, DivOp, FillOp, MulOp};
use bolt_core::{Backend, Tensor};
use num_traits::cast;
use tinyvec::ArrayVec;

use crate::Float;
use crate::backward::{BackwardContext, BackwardOp, MAX_INPUTS};
use crate::error::Result;

pub struct ReluBackward;

impl ReluBackward {
    pub fn new() -> Self {
        Self
    }
}

impl<B, D> BackwardOp<B, D> for ReluBackward
where
    B: Backend<D> + AddOp<D> + MulOp<D> + DivOp<D> + AbsOp<D> + FillOp<D> + CopyOp<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>> {
        let input = ctx.saved(0);

        let epsilon: D = cast(1e-8).unwrap();
        let eps_tensor = Tensor::full(&input.backend(), input.shape(), epsilon)?;
        let abs_input = input.abs()?;
        let sign_denom = abs_input.add(&eps_tensor)?;
        let sign_like = input.div(&sign_denom)?;

        let one = Tensor::full(&input.backend(), input.shape(), cast::<_, D>(1.0).unwrap())?;
        let sign_plus_one = sign_like.add(&one)?;
        let two: D = cast(2.0).unwrap();
        let two_tensor = Tensor::full(&input.backend(), input.shape(), two)?;
        let mask = sign_plus_one.div(&two_tensor)?;

        let grad_input = grad_output.mul(&mask)?;

        let mut result = ArrayVec::new();
        result.push(Some(grad_input));
        Ok(result)
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}
