use bolt_core::Tensor;
use bolt_core::backend::{Backend, ReluOp};
use bolt_core::dtype::FloatType;

use crate::context::Context;
use crate::error::Result;
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

impl<B, D> Model<B, D> for ReLU
where
    B: Backend + ReluOp<D>,
    D: FloatType,
{
    type Input = Tensor<B, D>;
    type Output = Result<Tensor<B, D>>;

    fn forward(&self, _ctx: &Context<B>, input: Self::Input) -> Self::Output {
        Ok(input.relu()?)
    }
}

pub fn relu() -> ReLU {
    ReLU::new()
}
