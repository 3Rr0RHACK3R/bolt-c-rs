//! Adam optimizer with decoupled weight decay (AdamW).
//!
//! Reference: https://arxiv.org/abs/1412.6980 (Adam)
//! Reference: https://arxiv.org/abs/1711.05101 (AdamW / Decoupled Weight Decay)

use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use bolt_core::backend::{AddOp, CopyOp, DivOp, FillOp, MulOp, NegOp, PowOp, ReshapeOp, SqrtOp, SubOp, SumOp};
use bolt_core::{BaseBackend, Error, Float, Result};
use bolt_nn::Param;
use bolt_tensor::{Tensor, no_grad};

/// Configuration for the Adam optimizer.
#[derive(Clone, Copy, Debug)]
pub struct AdamCfg {
    /// Learning rate (default: 1e-3)
    pub lr: f64,
    /// Coefficients for computing running averages of gradient and its square (default: (0.9, 0.999))
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8)
    pub eps: f64,
    /// Decoupled weight decay (default: 0.0)
    pub weight_decay: f64,
}

impl Default for AdamCfg {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Per-group configuration overrides for Adam.
#[derive(Clone, Copy, Debug)]
pub struct AdamGroupCfg {
    /// Multiplier for base learning rate (default: 1.0)
    pub lr_mult: f64,
    /// Optional override for weight decay (uses base if None)
    pub weight_decay: Option<f64>,
}

impl Default for AdamGroupCfg {
    fn default() -> Self {
        Self {
            lr_mult: 1.0,
            weight_decay: None,
        }
    }
}

/// First moment (mean) buffer for a parameter.
struct MomentState<B: BaseBackend, D: Float> {
    /// First moment estimate (exponential moving average of gradients)
    m: Tensor<B, D>,
    /// Second moment estimate (exponential moving average of squared gradients)
    v: Tensor<B, D>,
}

/// Adam optimizer with optional decoupled weight decay.
///
/// Update rule for each parameter `w` with gradient `g`:
/// ```text
/// m_t = β₁ * m_{t-1} + (1 - β₁) * g
/// v_t = β₂ * v_{t-1} + (1 - β₂) * g²
/// m̂_t = m_t / (1 - β₁^t)   // bias correction
/// v̂_t = v_t / (1 - β₂^t)   // bias correction
/// w_t = w_{t-1} - lr * m̂_t / (√v̂_t + ε) - lr * wd * w_{t-1}
/// ```
pub struct Adam<B, D>
where
    B: BaseBackend,
    D: Float,
{
    backend: Arc<B>,
    base: AdamCfg,
    group: BTreeMap<u32, AdamGroupCfg>,
    state: BTreeMap<String, MomentState<B, D>>,
    step_count: u64,
}

impl<B, D> Adam<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(backend: Arc<B>, cfg: AdamCfg) -> Result<Self> {
        validate_cfg(&cfg)?;
        Ok(Self {
            backend,
            base: cfg,
            group: BTreeMap::new(),
            state: BTreeMap::new(),
            step_count: 0,
        })
    }

    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    /// Set configuration overrides for a parameter group.
    pub fn set_group(&mut self, group_id: u32, cfg: AdamGroupCfg) -> Result<()> {
        validate_group_cfg(&cfg)?;
        self.group.insert(group_id, cfg);
        Ok(())
    }

    /// Get the current step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Perform one optimization step.
    pub fn step(&mut self, params: &[Param<B, D>]) -> Result<()>
    where
        B: CopyOp<D>
            + AddOp<D>
            + SubOp<D>
            + MulOp<D>
            + DivOp<D>
            + FillOp<D>
            + ReshapeOp<D>
            + SumOp<D>
            + NegOp<D>
            + SqrtOp<D>
            + PowOp<D>
            + 'static,
        D: Float + 'static,
    {
        let _ng = no_grad();

        self.step_count += 1;
        let t = self.step_count as f64;

        let (beta1, beta2) = self.base.betas;
        let eps = self.base.eps;
        let backend = &self.backend;

        // Pre-compute scalar tensors that are constant for all parameters.
        // This avoids redundant allocations when updating many parameters.
        let beta1_t = Tensor::from_slice(backend, &[D::from_f64(beta1)], &[])?;
        let beta2_t = Tensor::from_slice(backend, &[D::from_f64(beta2)], &[])?;
        let one_minus_beta1 = Tensor::from_slice(backend, &[D::from_f64(1.0 - beta1)], &[])?;
        let one_minus_beta2 = Tensor::from_slice(backend, &[D::from_f64(1.0 - beta2)], &[])?;
        let eps_t = Tensor::from_slice(backend, &[D::from_f64(eps)], &[])?;

        // Bias correction factors (depend on step count, constant per step)
        let bias_corr1 = 1.0 - beta1.powi(t as i32);
        let bias_corr2 = 1.0 - beta2.powi(t as i32);
        let bias_corr1_t = Tensor::from_slice(backend, &[D::from_f64(bias_corr1)], &[])?;
        let bias_corr2_t = Tensor::from_slice(backend, &[D::from_f64(bias_corr2)], &[])?;

        // Cache for per-group lr/wd tensors to avoid recreating for params in the same group
        let mut lr_cache: BTreeMap<u32, Tensor<B, D>> = BTreeMap::new();
        let mut wd_cache: BTreeMap<u32, Option<Tensor<B, D>>> = BTreeMap::new();

        let mut seen = HashSet::new();

        for p in params {
            let key = p.key().to_string();
            if !seen.insert(key.clone()) {
                continue;
            }
            if !p.requires_grad() {
                continue;
            }

            let Some(g) = p.grad() else {
                continue;
            };

            if g.shape() != p.shape() {
                return Err(Error::ShapeMismatch {
                    lhs: g.shape().to_vec(),
                    rhs: p.shape().to_vec(),
                });
            }

            let group_id = p.group();
            let (lr_val, wd_val) = self.cfg_for(group_id);

            let w = p.tensor();

            // Initialize or get moment buffers
            let state = match self.state.entry(key) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(MomentState {
                    m: Tensor::zeros_like(&w)?,
                    v: Tensor::zeros_like(&w)?,
                }),
            };

            // Reinitialize if shapes changed
            if state.m.shape() != w.shape() {
                state.m = Tensor::zeros_like(&w)?;
                state.v = Tensor::zeros_like(&w)?;
            }

            // Get or create cached lr tensor for this group
            let lr_t = match lr_cache.entry(group_id) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(Tensor::from_slice(backend, &[D::from_f64(lr_val)], &[])?),
            };

            // m_t = β₁ * m_{t-1} + (1 - β₁) * g
            let m_new = state.m.mul(&beta1_t)?.add(&g.mul(&one_minus_beta1)?)?;

            // v_t = β₂ * v_{t-1} + (1 - β₂) * g²
            let g_sq = g.mul(&g)?;
            let v_new = state.v.mul(&beta2_t)?.add(&g_sq.mul(&one_minus_beta2)?)?;

            // Bias-corrected estimates: m̂ = m / (1 - β₁^t), v̂ = v / (1 - β₂^t)
            let m_hat = m_new.div(&bias_corr1_t)?;
            let v_hat = v_new.div(&bias_corr2_t)?;

            // Update: w = w - lr * m̂ / (√v̂ + ε)
            let sqrt_v = v_hat.sqrt()?;
            let denom = sqrt_v.add(&eps_t)?;
            let step_dir = m_hat.div(&denom)?;
            let update = step_dir.mul(lr_t)?;
            let mut w_new = w.sub(&update)?;

            // Apply decoupled weight decay: w = w - lr * wd * w
            if wd_val > 0.0 {
                // Get or create cached wd tensor for this group
                let wd_factor = match wd_cache.entry(group_id) {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => e.insert(Some(
                        Tensor::from_slice(backend, &[D::from_f64(lr_val * wd_val)], &[])?
                    )),
                };
                if let Some(wd_t) = wd_factor {
                    let wd_term = w.mul(wd_t)?;
                    w_new = w_new.sub(&wd_term)?;
                }
            }

            // Store updated moment buffers
            state.m = m_new;
            state.v = v_new;

            p.set_tensor(w_new)
                .map_err(|e| Error::OpError(format!("{e:?}")))?;
        }

        Ok(())
    }

    fn cfg_for(&self, group_id: u32) -> (f64, f64) {
        let cfg = self.group.get(&group_id).copied().unwrap_or_default();
        let lr = self.base.lr * cfg.lr_mult;
        let wd = cfg.weight_decay.unwrap_or(self.base.weight_decay);
        (lr, wd)
    }

    /// Access first moment (m) state for inspection or serialization.
    pub fn first_moment_state(&self) -> BTreeMap<String, &Tensor<B, D>> {
        self.state.iter().map(|(k, v)| (k.clone(), &v.m)).collect()
    }

    /// Access second moment (v) state for inspection or serialization.
    pub fn second_moment_state(&self) -> BTreeMap<String, &Tensor<B, D>> {
        self.state.iter().map(|(k, v)| (k.clone(), &v.v)).collect()
    }

    /// Mutable access to first moment (m) state for loading checkpoints.
    pub fn first_moment_state_mut(&mut self) -> impl Iterator<Item = (&String, &mut Tensor<B, D>)> {
        self.state.iter_mut().map(|(k, v)| (k, &mut v.m))
    }

    /// Mutable access to second moment (v) state for loading checkpoints.
    pub fn second_moment_state_mut(&mut self) -> impl Iterator<Item = (&String, &mut Tensor<B, D>)> {
        self.state.iter_mut().map(|(k, v)| (k, &mut v.v))
    }

    /// Reset optimizer state (moments and step count).
    pub fn reset(&mut self) {
        self.state.clear();
        self.step_count = 0;
    }

    /// Set the step count (useful when loading checkpoints).
    pub fn set_step_count(&mut self, count: u64) {
        self.step_count = count;
    }
}

fn validate_cfg(cfg: &AdamCfg) -> Result<()> {
    if cfg.lr <= 0.0 {
        return Err(Error::OpError(format!(
            "adam: lr must be positive, got {}",
            cfg.lr
        )));
    }
    if cfg.betas.0 < 0.0 || cfg.betas.0 >= 1.0 {
        return Err(Error::OpError(format!(
            "adam: beta1 must be in [0, 1), got {}",
            cfg.betas.0
        )));
    }
    if cfg.betas.1 < 0.0 || cfg.betas.1 >= 1.0 {
        return Err(Error::OpError(format!(
            "adam: beta2 must be in [0, 1), got {}",
            cfg.betas.1
        )));
    }
    if cfg.eps <= 0.0 {
        return Err(Error::OpError(format!(
            "adam: eps must be positive, got {}",
            cfg.eps
        )));
    }
    if cfg.weight_decay < 0.0 {
        return Err(Error::OpError(format!(
            "adam: weight_decay must be non-negative, got {}",
            cfg.weight_decay
        )));
    }
    Ok(())
}

fn validate_group_cfg(cfg: &AdamGroupCfg) -> Result<()> {
    if cfg.lr_mult <= 0.0 {
        return Err(Error::OpError(format!(
            "adam: lr_mult must be positive, got {}",
            cfg.lr_mult
        )));
    }
    if let Some(wd) = cfg.weight_decay
        && wd < 0.0
    {
        return Err(Error::OpError(format!(
            "adam: group weight_decay must be non-negative, got {}",
            wd
        )));
    }
    Ok(())
}
