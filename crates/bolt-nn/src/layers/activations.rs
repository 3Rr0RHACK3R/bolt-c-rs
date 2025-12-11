use bolt_autodiff::{Float, HasParams, Parameter};
use bolt_core::BaseBackend;
use bolt_core::Tensor;
use bolt_core::backend::{Backend, ReluOp};

use crate::context::Context;
use crate::error::Result;
use crate::mode::Mode;
use crate::model::Model;

pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D> HasParams<B, D> for ReLU
where
    B: BaseBackend,
    D: Float,
{
    fn params(&self) -> Vec<&Parameter<B, D>> {
        vec![]
    }

    fn params_mut(&mut self) -> Vec<&mut Parameter<B, D>> {
        vec![]
    }
}

impl<B, D, M> Model<B, D, M> for ReLU
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
    M::Backend: Backend + ReluOp<D>,
{
    type Input = Tensor<M::Backend, D>;
    type Output = Result<Tensor<M::Backend, D>>;

    fn forward(&self, _ctx: &Context<B, D, M>, input: Self::Input) -> Self::Output {
        Ok(input.relu()?)
    }
}

pub fn relu() -> ReLU {
    ReLU::new()
}
