use bolt_core::BaseBackend;
use bolt_core::Float;
use bolt_core::backend::CopyOp;
use bolt_core::backend::{AddOp, MatmulOp, ReshapeOp, SumOp, TransposeOp};
use bolt_tensor::Tensor;

use crate::{Error, ForwardCtx, Init, Module, Param, Result, Store};

pub struct Linear<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub weight: Param<B, D>,
    pub bias: Option<Param<B, D>>,
    in_features: usize,
    out_features: usize,
}

impl<B, D> Linear<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn init(
        store: &Store<B, D>,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = store.param(
            "weight",
            &[out_features, in_features],
            Init::KaimingUniform {
                a: D::from_f64(0.0),
            },
        )?;

        let bias = if bias {
            Some(store.param("bias", &[out_features], Init::Zeros)?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias,
            in_features,
            out_features,
        })
    }
}

impl<B, D> Module<B, D> for Linear<B, D>
where
    B: BaseBackend
        + AddOp<D>
        + CopyOp<D>
        + MatmulOp<D>
        + ReshapeOp<D>
        + SumOp<D>
        + TransposeOp<D>
        + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        let shape = x.shape();
        if shape.len() != 2 || shape[1] != self.in_features {
            return Err(Error::Shape(format!(
                "Linear: expected [batch, {}], got {:?}",
                self.in_features, shape
            )));
        }

        let w = self.weight.tensor();
        let wt = w.transpose(-1, -2)?;
        let mut y = x.matmul(&wt)?;

        if let Some(b) = &self.bias {
            let b = b.tensor();
            y = y.add(&b)?;
        }

        if y.shape()[1] != self.out_features {
            return Err(Error::Shape(format!(
                "Linear: expected out_features {}, got {:?}",
                self.out_features,
                y.shape()
            )));
        }

        Ok(y)
    }
}
