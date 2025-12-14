use std::collections::{HashMap, HashSet};

use bolt_autodiff::{Float, ParamId, Parameter};
use bolt_core::backend::{AddOp, FillOp, MulOp, SubOp};
use bolt_core::{BaseBackend, Tensor};

use crate::error::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct SgdConfig {
    pub learning_rate: f64,
    pub momentum: Option<f64>,
    pub weight_decay: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct Sgd<B, D>
where
    B: BaseBackend,
    D: Float + std::fmt::Display,
{
    pub(crate) config: SgdConfig,
    pub(crate) state: SgdState<B, D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SgdBuilder {
    learning_rate: f64,
    momentum: Option<f64>,
    weight_decay: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct SgdParamState<B, D>
where
    B: BaseBackend,
    D: Float + std::fmt::Display,
{
    pub(crate) velocity: Option<Tensor<B, D>>,
}

#[derive(Clone, Debug)]
pub struct SgdState<B, D>
where
    B: BaseBackend,
    D: Float + std::fmt::Display,
{
    pub(crate) per_param: HashMap<ParamId, SgdParamState<B, D>>,
    pub(crate) step: u64,
}

impl SgdBuilder {
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: None,
            weight_decay: None,
        }
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn momentum(mut self, m: f64) -> Self {
        self.momentum = Some(m);
        self
    }

    pub fn weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = Some(wd);
        self
    }

    pub fn init<B, D>(self, params: &[&Parameter<B, D>]) -> Result<Sgd<B, D>>
    where
        B: BaseBackend + FillOp<D>,
        D: Float + std::fmt::Display,
    {
        validate_hyperparams(self.learning_rate, self.momentum, self.weight_decay)?;

        let mut per_param = HashMap::new();
        let mut seen = HashSet::new();

        for param in params {
            if !seen.insert(param.id()) {
                continue;
            }

            let mut state = SgdParamState { velocity: None };

            if self.momentum.is_some() {
                state.velocity = Some(Tensor::zeros_like(param.tensor())?);
            }

            per_param.insert(param.id(), state);
        }

        Ok(Sgd {
            config: SgdConfig {
                learning_rate: self.learning_rate,
                momentum: self.momentum,
                weight_decay: self.weight_decay,
            },
            state: SgdState { per_param, step: 0 },
        })
    }
}

impl<B, D> Sgd<B, D>
where
    B: BaseBackend + AddOp<D> + SubOp<D> + MulOp<D> + FillOp<D>,
    D: Float + std::fmt::Display,
{
    pub fn builder() -> SgdBuilder {
        SgdBuilder::new()
    }

    pub fn config(&self) -> &SgdConfig {
        &self.config
    }

    pub fn state(&self) -> &SgdState<B, D> {
        &self.state
    }

    pub fn step(&mut self, params: &mut [&mut Parameter<B, D>]) -> Result<()> {
        let lr_value = cast_scalar::<D, _>(self.config.learning_rate, |value| {
            Error::InvalidLearningRate { value }
        })?;
        let momentum_value = self
            .config
            .momentum
            .map(|m| cast_scalar::<D, _>(m, |value| Error::InvalidMomentum { value }))
            .transpose()?;
        let weight_decay_value = self
            .config
            .weight_decay
            .map(|wd| cast_scalar::<D, _>(wd, |value| Error::InvalidWeightDecay { value }))
            .transpose()?;

        let mut seen = HashSet::new();

        for param in params.iter_mut() {
            if !seen.insert(param.id()) {
                continue;
            }

            self.update_parameter(param, lr_value, momentum_value, weight_decay_value)?;
        }

        self.state.step = self.state.step.saturating_add(1);
        Ok(())
    }

    fn update_parameter(
        &mut self,
        param: &mut Parameter<B, D>,
        lr: D,
        momentum: Option<D>,
        weight_decay: Option<D>,
    ) -> Result<()> {
        let state =
            self.state
                .per_param
                .get_mut(&param.id())
                .ok_or_else(|| Error::UnknownParameter {
                    param_id: param.id(),
                    param_name: param.name().map(|n| n.to_string()),
                })?;

        let grad = param.grad().ok_or_else(|| Error::MissingGradient {
            param_id: param.id(),
            param_name: param.name().map(|n| n.to_string()),
        })?;

        let grad_shape = grad.shape().to_vec();
        let param_shape = param.tensor().shape().to_vec();
        if grad_shape != param_shape {
            return Err(Error::ShapeMismatch {
                param_id: param.id(),
                param_name: param.name().map(|n| n.to_string()),
                grad_shape,
                param_shape,
            });
        }

        let mut grad_tensor = grad.clone();

        if let Some(wd) = weight_decay {
            grad_tensor = Self::apply_weight_decay(param.tensor(), &grad_tensor, wd)?;
        }

        let update = if let Some(mom) = momentum {
            Self::apply_momentum(state, param.tensor(), &grad_tensor, mom)?
        } else {
            grad_tensor
        };

        let lr_tensor = scalar_tensor(param.tensor(), lr)?;
        let scaled_update = update.mul(&lr_tensor)?;
        let new_value = param.tensor().sub(&scaled_update)?;
        *param.tensor_mut() = new_value;

        Ok(())
    }

    fn apply_weight_decay(
        param_tensor: &Tensor<B, D>,
        grad: &Tensor<B, D>,
        wd: D,
    ) -> Result<Tensor<B, D>> {
        let wd_tensor = scalar_tensor(param_tensor, wd)?;
        let decay = param_tensor.mul(&wd_tensor)?;
        Ok(grad.add(&decay)?)
    }

    fn apply_momentum(
        state: &mut SgdParamState<B, D>,
        param_tensor: &Tensor<B, D>,
        grad: &Tensor<B, D>,
        momentum: D,
    ) -> Result<Tensor<B, D>> {
        let momentum_tensor = scalar_tensor(param_tensor, momentum)?;
        let velocity = match state.velocity.take() {
            Some(v) => v,
            None => Tensor::zeros_like(param_tensor)?,
        };
        let updated_velocity = velocity.mul(&momentum_tensor)?.add(grad)?;
        state.velocity = Some(updated_velocity.clone());
        Ok(updated_velocity)
    }
}

fn validate_hyperparams(lr: f64, momentum: Option<f64>, weight_decay: Option<f64>) -> Result<()> {
    if lr <= 0.0 {
        return Err(Error::InvalidLearningRate { value: lr });
    }
    if let Some(m) = momentum {
        if m < 0.0 {
            return Err(Error::InvalidMomentum { value: m });
        }
    }
    if let Some(wd) = weight_decay {
        if wd < 0.0 {
            return Err(Error::InvalidWeightDecay { value: wd });
        }
    }
    Ok(())
}

fn cast_scalar<D, F>(value: f64, _err: F) -> Result<D>
where
    D: Float,
    F: FnOnce(f64) -> Error,
{
    Ok(D::from_f64(value))
}

fn scalar_tensor<B, D>(like: &Tensor<B, D>, value: D) -> Result<Tensor<B, D>>
where
    B: BaseBackend,
    D: Float + std::fmt::Display,
{
    Ok(Tensor::from_slice(&like.backend(), &[value], &[])?)
}
