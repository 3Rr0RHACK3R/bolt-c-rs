use bolt_core::backend::{AddOp, Backend, FillOp, MatmulOp, TransposeOp};
use bolt_core::dtype::FloatType;
use bolt_core::Tensor;
use serde::{Deserialize, Serialize};

use crate::context::Context;
use crate::error::Result;
use crate::model::Model;

#[derive(Clone, Serialize, Deserialize)]
pub struct LinearSpec {
    pub in_features: usize,
    pub out_features: usize,
    pub bias: bool,
}

impl LinearSpec {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            bias: true,
        }
    }

    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn build<B, D>(&self, ctx: &Context<B>) -> Result<Linear<B, D>>
    where
        B: Backend + FillOp<D>,
        D: FloatType,
    {
        let weight = Tensor::full(
            ctx.backend(),
            &[self.out_features, self.in_features],
            D::default(),
        )?;

        let bias = if self.bias {
            Some(Tensor::full(ctx.backend(), &[self.out_features], D::default())?)
        } else {
            None
        };

        Ok(Linear { weight, bias })
    }
}

pub struct Linear<B, D>
where
    B: Backend,
    D: bolt_core::dtype::NativeType,
{
    pub weight: Tensor<B, D>,
    pub bias: Option<Tensor<B, D>>,
}

impl<B, D> Model<B, D> for Linear<B, D>
where
    B: Backend + MatmulOp<D> + AddOp<D> + TransposeOp<D>,
    D: FloatType,
{
    type Input = Tensor<B, D>;
    type Output = Result<Tensor<B, D>>;

    fn forward(&self, _ctx: &Context<B>, input: Self::Input) -> Self::Output {
        let weight_t = self.weight.transpose(-1, -2)?;
        let out = input.matmul(&weight_t)?;

        match &self.bias {
            Some(bias) => Ok(out.add(bias)?),
            None => Ok(out),
        }
    }
}

pub fn linear(in_features: usize, out_features: usize) -> LinearSpec {
    LinearSpec::new(in_features, out_features)
}
