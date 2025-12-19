use bolt_autodiff::{Autodiff, Float, HasParams};
use bolt_core::BaseBackend;
use bolt_core::Tensor;

use crate::error::Result;
use crate::mode::{Eval, Grad};
use crate::model::Model;
use crate::run_mode::Trainable;
use crate::Context;

pub trait DualModel<B, D>: HasParams<B, D> + Trainable + Send + Sync
where
    B: BaseBackend,
    D: Float,
{
    fn forward_eval(
        &self,
        ctx: &Context<B, D, Eval<B, D>>,
        input: Tensor<B, D>,
    ) -> Result<Tensor<B, D>>;

    fn forward_grad(
        &self,
        ctx: &Context<B, D, Grad<B, D>>,
        input: Tensor<Autodiff<B, D>, D>,
    ) -> Result<Tensor<Autodiff<B, D>, D>>;
}

impl<T, B, D> DualModel<B, D> for T
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D>
        + Trainable
        + Model<
            B,
            D,
            Eval<B, D>,
            Input = Tensor<B, D>,
            Output = Result<Tensor<B, D>>,
        >
        + Model<
            B,
            D,
            Grad<B, D>,
            Input = Tensor<Autodiff<B, D>, D>,
            Output = Result<Tensor<Autodiff<B, D>, D>>,
        >
        + Send
        + Sync,
{
    fn forward_eval(
        &self,
        ctx: &Context<B, D, Eval<B, D>>,
        input: Tensor<B, D>,
    ) -> Result<Tensor<B, D>> {
        <T as Model<B, D, Eval<B, D>>>::forward(self, ctx, input)
    }

    fn forward_grad(
        &self,
        ctx: &Context<B, D, Grad<B, D>>,
        input: Tensor<Autodiff<B, D>, D>,
    ) -> Result<Tensor<Autodiff<B, D>, D>> {
        <T as Model<B, D, Grad<B, D>>>::forward(self, ctx, input)
    }
}
