use std::sync::Arc;

use bolt_autodiff::{Float, HasParams, Parameter};
use bolt_core::BaseBackend;
use bolt_core::Tensor;
use bolt_core::backend::{AddOp, Backend, FillOp, MatmulOp, TransposeOp};
use serde::{Deserialize, Serialize};

use crate::context::Context;
use crate::error::Result;
use crate::mode::Mode;
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

    pub fn build<B, D>(&self, backend: &Arc<B>) -> Result<Linear<B, D>>
    where
        B: BaseBackend + FillOp<D>,
        D: Float,
    {
        let weight_tensor = Tensor::full(
            backend,
            &[self.out_features, self.in_features],
            D::default(),
        )?;
        let weight = Parameter::with_name(weight_tensor, "weight");

        let bias = if self.bias {
            let bias_tensor = Tensor::full(backend, &[self.out_features], D::default())?;
            Some(Parameter::with_name(bias_tensor, "bias"))
        } else {
            None
        };

        Ok(Linear { weight, bias })
    }
}

pub struct Linear<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub weight: Parameter<B, D>,
    pub bias: Option<Parameter<B, D>>,
}

impl<B, D> HasParams<B, D> for Linear<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn params(&self) -> Vec<&Parameter<B, D>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn params_mut(&mut self) -> Vec<&mut Parameter<B, D>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }
}

impl<B, D, M> Model<B, D, M> for Linear<B, D>
where
    B: BaseBackend + MatmulOp<D> + AddOp<D> + TransposeOp<D>,
    D: Float,
    M: Mode<B, D>,
    M::Backend: Backend + MatmulOp<D> + AddOp<D> + TransposeOp<D>,
{
    type Input = Tensor<M::Backend, D>;
    type Output = Result<Tensor<M::Backend, D>>;

    fn forward(&self, ctx: &Context<B, D, M>, input: Self::Input) -> Self::Output {
        let weight = ctx.param(&self.weight);
        let weight_t = weight.transpose(-1, -2)?;
        let out = input.matmul(&weight_t)?;

        match &self.bias {
            Some(bias) => {
                let bias_tensor = ctx.param(bias);
                Ok(out.add(&bias_tensor)?)
            }
            None => Ok(out),
        }
    }
}

pub fn linear(in_features: usize, out_features: usize) -> LinearSpec {
    LinearSpec::new(in_features, out_features)
}
