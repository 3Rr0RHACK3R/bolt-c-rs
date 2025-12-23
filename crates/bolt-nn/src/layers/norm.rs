use bolt_core::backend::{
    AddOp, BroadcastToOp, CopyOp, DivOp, FillOp, MeanOp, MulOp, NegOp, ReshapeOp, SqueezeOp,
    SqrtOp, SubOp, SumOp, UnsqueezeOp,
};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::{Tensor, no_grad};

use crate::{Buffer, Error, ForwardCtx, Init, Module, Param, Result, Store};

pub struct Norm<B, D>
where
    B: BaseBackend,
    D: Float,
{
    gamma: Option<Param<B, D>>,
    beta: Option<Param<B, D>>,
    running_mean: Option<Buffer<B, D>>,
    running_var: Option<Buffer<B, D>>,
    axes: Vec<isize>,
    normalized_shape: Vec<usize>,
    eps: f64,
    momentum: f64,
}

pub struct NormConfig {
    pub axes: Vec<isize>,
    pub normalized_shape: Vec<usize>,
    pub affine: bool,
    pub track_running_stats: bool,
    pub eps: f64,
    pub momentum: f64,
}

impl Default for NormConfig {
    fn default() -> Self {
        Self {
            axes: vec![],
            normalized_shape: vec![],
            affine: true,
            track_running_stats: false,
            eps: 1e-5,
            momentum: 0.1,
        }
    }
}

impl<B, D> Norm<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn init(store: &Store<B, D>, config: NormConfig) -> Result<Self> {
        if config.normalized_shape.is_empty() {
            return Err(Error::Shape(
                "Norm: normalized_shape must not be empty".into(),
            ));
        }
        if config.axes.is_empty() {
            return Err(Error::Shape("Norm: axes must not be empty".into()));
        }

        let gamma = if config.affine {
            Some(store.param("gamma", &config.normalized_shape, Init::Ones)?)
        } else {
            None
        };

        let beta = if config.affine {
            Some(store.param("beta", &config.normalized_shape, Init::Zeros)?)
        } else {
            None
        };

        let running_mean = if config.track_running_stats {
            Some(store.buffer("running_mean", &config.normalized_shape, Init::Zeros)?)
        } else {
            None
        };

        let running_var = if config.track_running_stats {
            Some(store.buffer("running_var", &config.normalized_shape, Init::Ones)?)
        } else {
            None
        };

        Ok(Self {
            gamma,
            beta,
            running_mean,
            running_var,
            axes: config.axes,
            normalized_shape: config.normalized_shape,
            eps: config.eps,
            momentum: config.momentum,
        })
    }

    pub fn axes(&self) -> &[isize] {
        &self.axes
    }

    pub fn normalized_shape(&self) -> &[usize] {
        &self.normalized_shape
    }

    pub fn eps(&self) -> f64 {
        self.eps
    }

    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    pub fn track_running_stats(&self) -> bool {
        self.running_mean.is_some()
    }

    pub fn gamma(&self) -> Option<&Param<B, D>> {
        self.gamma.as_ref()
    }

    pub fn beta(&self) -> Option<&Param<B, D>> {
        self.beta.as_ref()
    }

    pub fn running_mean(&self) -> Option<&Buffer<B, D>> {
        self.running_mean.as_ref()
    }

    pub fn running_var(&self) -> Option<&Buffer<B, D>> {
        self.running_var.as_ref()
    }
}

impl<B, D> Module<B, D> for Norm<B, D>
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
        let axes: Vec<isize> = self.axes.clone();
        let axes_slice = axes.as_slice();

        let (mean, var) = if ctx.is_train() || !self.track_running_stats() {
            let mean = x.mean(Some(axes_slice), true)?;
            let centered = x.sub(&mean)?;
            let var = centered.mul(&centered)?.mean(Some(axes_slice), true)?;

            if ctx.is_train() {
                if let (Some(rm), Some(rv)) = (&self.running_mean, &self.running_var) {
                    let _guard = no_grad();

                    let batch_mean = mean.detach().squeeze()?;
                    let batch_var = var.detach().squeeze()?;

                    let old_mean = rm.tensor();
                    let old_var = rv.tensor();

                    let m = D::from_f64(self.momentum);
                    let one_minus_m = D::from_f64(1.0 - self.momentum);

                    let m_tensor = Tensor::<B, D>::full_like(&batch_mean, m)?;
                    let one_minus_m_tensor = Tensor::<B, D>::full_like(&batch_mean, one_minus_m)?;

                    let new_mean = old_mean
                        .mul(&one_minus_m_tensor)?
                        .add(&batch_mean.mul(&m_tensor)?)?;
                    let new_var = old_var
                        .mul(&one_minus_m_tensor)?
                        .add(&batch_var.mul(&m_tensor)?)?;

                    rm.set(new_mean)?;
                    rv.set(new_var)?;
                }
            }

            (mean, var)
        } else {
            let rm = self
                .running_mean
                .as_ref()
                .ok_or_else(|| Error::State("Norm: missing running_mean in eval mode".into()))?;
            let rv = self
                .running_var
                .as_ref()
                .ok_or_else(|| Error::State("Norm: missing running_var in eval mode".into()))?;

            let mean = expand_to_input_shape(&rm.tensor(), x.shape())?;
            let var = expand_to_input_shape(&rv.tensor(), x.shape())?;

            (mean, var)
        };

        let eps_tensor = Tensor::<B, D>::full_like(&var, D::from_f64(self.eps))?;
        let std = var.add(&eps_tensor)?.sqrt()?;
        let x_hat = x.sub(&mean)?.div(&std)?;

        let y = if let (Some(gamma), Some(beta)) = (&self.gamma, &self.beta) {
            let gamma_expanded = expand_to_input_shape(&gamma.tensor(), x_hat.shape())?;
            let beta_expanded = expand_to_input_shape(&beta.tensor(), x_hat.shape())?;
            x_hat.mul(&gamma_expanded)?.add(&beta_expanded)?
        } else {
            x_hat
        };

        Ok(y)
    }
}

fn expand_to_input_shape<B, D>(param: &Tensor<B, D>, input_shape: &[usize]) -> Result<Tensor<B, D>>
where
    B: BaseBackend + BroadcastToOp<D> + CopyOp<D> + ReshapeOp<D> + SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    D: Float + 'static,
{
    let param_shape = param.shape();
    let input_rank = input_shape.len();
    let param_rank = param_shape.len();

    if param_rank == input_rank {
        return Ok(param.clone());
    }

    let mut expanded = param.clone();
    expanded = expanded.unsqueeze(0)?;

    while expanded.shape().len() < input_rank {
        let current_rank = expanded.shape().len() as isize;
        expanded = expanded.unsqueeze(current_rank)?;
    }

    Ok(expanded)
}
