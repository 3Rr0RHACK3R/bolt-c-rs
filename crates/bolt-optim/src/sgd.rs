use std::collections::{BTreeMap, HashSet};
use std::collections::btree_map::Entry;
use std::sync::Arc;

use bolt_core::backend::{AddOp, CopyOp, FillOp, MulOp, NegOp, ReshapeOp, SubOp, SumOp};
use bolt_core::{BaseBackend, Error, Float, Result};
use bolt_nn::Param;
use bolt_tensor::{Tensor, no_grad};

#[derive(Clone, Copy, Debug)]
pub struct SgdCfg {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SgdGroupCfg {
    pub lr_mult: f64,
    pub weight_decay: Option<f64>,
}

impl Default for SgdGroupCfg {
    fn default() -> Self {
        Self {
            lr_mult: 1.0,
            weight_decay: None,
        }
    }
}

pub struct Sgd<B, D>
where
    B: BaseBackend,
    D: Float,
{
    backend: Arc<B>,
    base: SgdCfg,
    group: BTreeMap<u32, SgdGroupCfg>,
    vel: BTreeMap<String, Tensor<B, D>>,
}

impl<B, D> Sgd<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(backend: Arc<B>, cfg: SgdCfg) -> Result<Self> {
        validate_cfg(cfg)?;
        Ok(Self {
            backend,
            base: cfg,
            group: BTreeMap::new(),
            vel: BTreeMap::new(),
        })
    }

    pub fn backend(&self) -> Arc<B> {
        self.backend.clone()
    }

    pub fn set_group(&mut self, group_id: u32, cfg: SgdGroupCfg) -> Result<()> {
        validate_group_cfg(cfg)?;
        self.group.insert(group_id, cfg);
        Ok(())
    }

    pub fn step(&mut self, params: &[Param<B, D>]) -> Result<()>
    where
        B: CopyOp<D>
            + AddOp<D>
            + SubOp<D>
            + MulOp<D>
            + FillOp<D>
            + ReshapeOp<D>
            + SumOp<D>
            + NegOp<D>
            + 'static,
        D: Float + 'static,
    {
        let _ng = no_grad();
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

            let (lr_val, wd_val) = self.cfg_for(p.group());

            let w = p.tensor();
            let backend = self.backend.clone();

            let v = match self.vel.entry(key) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    e.insert(Tensor::zeros_like(&w)?)
                }
            };
            if v.shape() != w.shape() {
                *v = Tensor::zeros_like(&w)?;
            }

            let lr_scalar = Tensor::from_slice(&backend, &[D::from_f64(lr_val)], &[])?;
            let mom_scalar = Tensor::from_slice(&backend, &[D::from_f64(self.base.momentum)], &[])?;
            let wd_scalar = Tensor::from_slice(&backend, &[D::from_f64(wd_val)], &[])?;

            let grad = g.add(&w.mul(&wd_scalar)?)?;
            let v_new = v.mul(&mom_scalar)?.add(&grad)?;
            let w_new = w.sub(&v_new.mul(&lr_scalar)?)?;

            *v = v_new;
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

    pub fn velocity_state(&self) -> &BTreeMap<String, Tensor<B, D>> {
        &self.vel
    }

    pub fn velocity_state_mut(&mut self) -> &mut BTreeMap<String, Tensor<B, D>> {
        &mut self.vel
    }
}

fn validate_cfg(cfg: SgdCfg) -> Result<()> {
    if !(cfg.lr > 0.0) {
        return Err(Error::OpError(format!(
            "sgd: lr must be positive, got {}",
            cfg.lr
        )));
    }
    if cfg.momentum < 0.0 {
        return Err(Error::OpError(format!(
            "sgd: momentum must be non-negative, got {}",
            cfg.momentum
        )));
    }
    if cfg.weight_decay < 0.0 {
        return Err(Error::OpError(format!(
            "sgd: weight_decay must be non-negative, got {}",
            cfg.weight_decay
        )));
    }
    Ok(())
}

fn validate_group_cfg(cfg: SgdGroupCfg) -> Result<()> {
    if !(cfg.lr_mult > 0.0) {
        return Err(Error::OpError(format!(
            "sgd: lr_mult must be positive, got {}",
            cfg.lr_mult
        )));
    }
    if let Some(wd) = cfg.weight_decay {
        if wd < 0.0 {
            return Err(Error::OpError(format!(
                "sgd: group weight_decay must be non-negative, got {}",
                wd
            )));
        }
    }
    Ok(())
}
