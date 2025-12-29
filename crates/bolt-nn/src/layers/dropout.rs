use bolt_core::backend::{BernoulliMaskOp, CopyOp, DivOp, FillOp, MulOp, NegOp, ReshapeOp, SumOp};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{Error, ForwardCtx, Module, Result};

pub struct Dropout {
    p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::State(format!(
                "Dropout: p must be in [0, 1], got {p}"
            )));
        }
        Ok(Self { p })
    }
}

impl<B, D> Module<B, D> for Dropout
where
    B: BaseBackend
        + BernoulliMaskOp<D>
        + CopyOp<D>
        + DivOp<D>
        + FillOp<D>
        + MulOp<D>
        + NegOp<D>
        + ReshapeOp<D>
        + SumOp<D>
        + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        if !ctx.is_train() || self.p == 0.0 {
            return Ok(x);
        }

        if self.p == 1.0 {
            return Ok(Tensor::zeros_like(&x)?);
        }

        let Some(rngs) = ctx.rngs_mut() else {
            return Err(Error::State(
                "Dropout requires RNG streams in training mode; use ForwardCtx::train_with_rngs"
                    .into(),
            ));
        };

        let keep_prob = 1.0 - self.p;
        let keep = D::from_f64(keep_prob);
        let seed = rngs.dropout.next_u64();

        let backend = x.backend();
        let mask =
            Tensor::<B, D>::bernoulli_mask(&backend, x.shape().as_slice(), keep, Some(seed))?;

        let y = x.mul(&mask)?;
        let keep_tensor = Tensor::<B, D>::full_like(&y, keep)?;
        Ok(y.div(&keep_tensor)?)
    }
}
