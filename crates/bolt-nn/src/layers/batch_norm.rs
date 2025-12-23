use bolt_core::backend::{
    AddOp, BroadcastToOp, CopyOp, DivOp, FillOp, MeanOp, MulOp, NegOp, ReshapeOp, SqueezeOp,
    SqrtOp, SubOp, SumOp, UnsqueezeOp,
};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::{Tensor, no_grad};

use crate::{Error, ForwardCtx, Module, Result, Store};

use super::norm::{Norm, NormConfig};

pub struct BatchNorm<B, D>
where
    B: BaseBackend,
    D: Float,
{
    inner: Norm<B, D>,
    num_features: usize,
}

impl<B, D> BatchNorm<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn init(
        store: &Store<B, D>,
        num_features: usize,
        affine: bool,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        let config = NormConfig {
            axes: vec![0],
            normalized_shape: vec![num_features],
            affine,
            track_running_stats: true,
            eps,
            momentum,
        };

        let inner = Norm::init(store, config)?;

        Ok(Self {
            inner,
            num_features,
        })
    }

    pub fn init_default(store: &Store<B, D>, num_features: usize) -> Result<Self> {
        Self::init(store, num_features, true, 1e-5, 0.1)
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f64 {
        self.inner.eps()
    }

    pub fn momentum(&self) -> f64 {
        self.inner.momentum()
    }

    fn reduction_axes(&self, rank: usize) -> Result<Vec<isize>> {
        match rank {
            2 => Ok(vec![0]),
            3 => Ok(vec![0, 2]),
            4 => Ok(vec![0, 2, 3]),
            5 => Ok(vec![0, 2, 3, 4]),
            _ => Err(Error::Shape(format!(
                "BatchNorm: unsupported input rank {}, expected 2-5",
                rank
            ))),
        }
    }
}

impl<B, D> Module<B, D> for BatchNorm<B, D>
where
    B: BaseBackend
        + AddOp<D>
        + BroadcastToOp<D>
        + CopyOp<D>
        + DivOp<D>
        + FillOp<D>
        + MeanOp<D>
        + MulOp<D>
        + NegOp<D>
        + ReshapeOp<D>
        + SqueezeOp<D>
        + SqrtOp<D>
        + SubOp<D>
        + SumOp<D>
        + UnsqueezeOp<D>
        + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        let shape = x.shape();
        let rank = shape.len();

        if rank < 2 {
            return Err(Error::Shape(format!(
                "BatchNorm: input must be at least 2D, got {}D",
                rank
            )));
        }

        if shape[1] != self.num_features {
            return Err(Error::Shape(format!(
                "BatchNorm: expected {} features at dim 1, got {}",
                self.num_features, shape[1]
            )));
        }

        let axes = self.reduction_axes(rank)?;
        let axes_slice = axes.as_slice();

        let (mean, var) = if ctx.is_train() {
            let mean = x.mean(Some(axes_slice), true)?;
            let centered = x.sub(&mean)?;
            let var = centered.mul(&centered)?.mean(Some(axes_slice), true)?;

            self.update_running_stats(&mean, &var)?;

            (mean, var)
        } else {
            self.get_running_stats_expanded(shape)?
        };

        let eps_tensor = Tensor::<B, D>::full_like(&var, D::from_f64(self.inner.eps()))?;
        let std = var.add(&eps_tensor)?.sqrt()?;
        let x_hat = x.sub(&mean)?.div(&std)?;

        self.apply_affine(x_hat)
    }
}

impl<B, D> BatchNorm<B, D>
where
    B: BaseBackend
        + AddOp<D>
        + BroadcastToOp<D>
        + CopyOp<D>
        + DivOp<D>
        + FillOp<D>
        + MeanOp<D>
        + MulOp<D>
        + NegOp<D>
        + ReshapeOp<D>
        + SqueezeOp<D>
        + SqrtOp<D>
        + SubOp<D>
        + SumOp<D>
        + UnsqueezeOp<D>
        + 'static,
    D: Float + 'static,
{
    fn update_running_stats(
        &self,
        batch_mean: &Tensor<B, D>,
        batch_var: &Tensor<B, D>,
    ) -> Result<()> {
        let _guard = no_grad();

        let batch_mean_sq = batch_mean.detach().squeeze()?;
        let batch_var_sq = batch_var.detach().squeeze()?;

        if let Some(rm) = self.inner.running_mean() {
            let old_mean = rm.tensor();
            let m = D::from_f64(self.inner.momentum());
            let one_minus_m = D::from_f64(1.0 - self.inner.momentum());

            let m_tensor = Tensor::<B, D>::full_like(&batch_mean_sq, m)?;
            let one_minus_m_tensor = Tensor::<B, D>::full_like(&batch_mean_sq, one_minus_m)?;

            let new_mean = old_mean
                .mul(&one_minus_m_tensor)?
                .add(&batch_mean_sq.mul(&m_tensor)?)?;

            rm.set(new_mean)?;
        }

        if let Some(rv) = self.inner.running_var() {
            let old_var = rv.tensor();
            let m = D::from_f64(self.inner.momentum());
            let one_minus_m = D::from_f64(1.0 - self.inner.momentum());

            let m_tensor = Tensor::<B, D>::full_like(&batch_var_sq, m)?;
            let one_minus_m_tensor = Tensor::<B, D>::full_like(&batch_var_sq, one_minus_m)?;

            let new_var = old_var
                .mul(&one_minus_m_tensor)?
                .add(&batch_var_sq.mul(&m_tensor)?)?;

            rv.set(new_var)?;
        }

        Ok(())
    }

    fn get_running_stats_expanded(
        &self,
        input_shape: &[usize],
    ) -> Result<(Tensor<B, D>, Tensor<B, D>)> {
        let rm = self
            .inner
            .running_mean()
            .ok_or_else(|| Error::State("BatchNorm: missing running_mean".into()))?;
        let rv = self
            .inner
            .running_var()
            .ok_or_else(|| Error::State("BatchNorm: missing running_var".into()))?;

        let mean = expand_to_input_shape(&rm.tensor(), input_shape)?;
        let var = expand_to_input_shape(&rv.tensor(), input_shape)?;

        Ok((mean, var))
    }

    fn apply_affine(&self, x_hat: Tensor<B, D>) -> Result<Tensor<B, D>> {
        match (self.inner.gamma(), self.inner.beta()) {
            (Some(gamma), Some(beta)) => {
                let gamma_exp = expand_to_input_shape(&gamma.tensor(), x_hat.shape())?;
                let beta_exp = expand_to_input_shape(&beta.tensor(), x_hat.shape())?;
                Ok(x_hat.mul(&gamma_exp)?.add(&beta_exp)?)
            }
            _ => Ok(x_hat),
        }
    }
}

fn expand_to_input_shape<B, D>(param: &Tensor<B, D>, input_shape: &[usize]) -> Result<Tensor<B, D>>
where
    B: BaseBackend + BroadcastToOp<D> + CopyOp<D> + ReshapeOp<D> + SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    D: Float + 'static,
{
    let input_rank = input_shape.len();
    let param_rank = param.shape().len();

    if param_rank == input_rank {
        return Ok(param.clone());
    }

    let mut expanded = param.unsqueeze(0)?;

    while expanded.shape().len() < input_rank {
        let current_rank = expanded.shape().len() as isize;
        expanded = expanded.unsqueeze(current_rank)?;
    }

    Ok(expanded)
}
