use std::collections::{BTreeMap, HashSet};

use bolt_core::backend::CopyOp;
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
    base: SgdCfg,
    group: BTreeMap<u32, SgdGroupCfg>,
    vel: BTreeMap<String, Vec<D>>,
    _marker: std::marker::PhantomData<B>,
}

impl<B, D> Sgd<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(cfg: SgdCfg) -> Result<Self> {
        validate_cfg(cfg)?;
        Ok(Self {
            base: cfg,
            group: BTreeMap::new(),
            vel: BTreeMap::new(),
            _marker: std::marker::PhantomData,
        })
    }

    pub fn set_group(&mut self, group_id: u32, cfg: SgdGroupCfg) -> Result<()> {
        validate_group_cfg(cfg)?;
        self.group.insert(group_id, cfg);
        Ok(())
    }

    pub fn step(&mut self, params: &[Param<B, D>]) -> Result<()>
    where
        B: CopyOp<D>,
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

            let (lr, wd) = self.cfg_for(p.group());

            let w = p.tensor();
            let backend = w.backend();
            let shape = w.shape().to_vec();

            let mut wv = w.to_vec()?;
            let gv = g.to_vec()?;

            let vbuf = self
                .vel
                .entry(key)
                .or_insert_with(|| vec![D::zero(); wv.len()]);
            if vbuf.len() != wv.len() {
                *vbuf = vec![D::zero(); wv.len()];
            }

            let lr = D::from_f64(lr);
            let mom = D::from_f64(self.base.momentum);
            let wd = D::from_f64(wd);

            for i in 0..wv.len() {
                let grad = gv[i] + wd * wv[i];
                vbuf[i] = mom * vbuf[i] + grad;
                wv[i] = wv[i] - lr * vbuf[i];
            }

            let updated = Tensor::from_vec(&backend, wv, &shape)?;
            p.set_tensor(updated)
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
