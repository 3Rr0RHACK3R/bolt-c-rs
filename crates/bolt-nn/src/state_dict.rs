use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use bolt_core::BaseBackend;
use bolt_core::Float;
use bolt_core::backend::CopyOp;

use crate::store::{Entry, Kind, Store};
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct TensorBlob<D: Float> {
    pub kind: Kind,
    pub group: u32,
    pub shape: Vec<usize>,
    pub data: Vec<D>,
}

#[derive(Clone, Debug)]
pub struct StateDict<D: Float> {
    pub format_version: u32,
    pub tensors: BTreeMap<String, TensorBlob<D>>,
    pub meta: BTreeMap<String, String>,
}

#[derive(Clone)]
pub struct LoadOptions {
    pub strict: bool,
    pub rename: Option<Arc<dyn Fn(&str) -> String + Send + Sync>>,
}

#[derive(Clone, Debug, Default)]
pub struct LoadReport {
    pub missing: Vec<String>,
    pub unexpected: Vec<String>,
    pub mismatched: Vec<(String, Vec<usize>, Vec<usize>)>,
}

impl<D: Float> StateDict<D> {
    pub fn new() -> Self {
        Self {
            format_version: 1,
            tensors: BTreeMap::new(),
            meta: BTreeMap::new(),
        }
    }
}

impl<B, D> Store<B, D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
{
    pub fn state_dict(&self) -> Result<StateDict<D>> {
        let (ps, bs) = self.all_entries();
        let mut sd = StateDict::new();

        for e in ps.into_iter().chain(bs.into_iter()) {
            let blob = entry_to_blob(&e)?;
            sd.tensors.insert(e.key.clone(), blob);
        }

        Ok(sd)
    }

    pub fn load_state_dict(&self, sd: &StateDict<D>, opt: LoadOptions) -> Result<LoadReport> {
        let rename = opt.rename;
        let mut used = BTreeSet::new();
        let mut report = LoadReport::default();

        for (k_in, blob) in sd.tensors.iter() {
            let k = match &rename {
                None => k_in.clone(),
                Some(f) => (f)(k_in),
            };

            match self.get_entry(&k) {
                None => report.unexpected.push(k_in.clone()),
                Some(e) => {
                    used.insert(k.clone());
                    if e.shape != blob.shape {
                        report
                            .mismatched
                            .push((k.clone(), e.shape.clone(), blob.shape.clone()));
                        continue;
                    }

                    let mut t = e.tensor.lock().unwrap();
                    let backend = t.backend();
                    let new_t =
                        bolt_tensor::Tensor::from_vec(&backend, blob.data.clone(), &blob.shape)?;

                    let requires_grad = e.requires_grad.load(std::sync::atomic::Ordering::Relaxed);
                    *t = new_t.with_requires_grad(requires_grad);
                }
            }
        }

        for k in self.expected_keys() {
            if !used.contains(&k) {
                report.missing.push(k);
            }
        }

        if opt.strict
            && (!report.missing.is_empty()
                || !report.unexpected.is_empty()
                || !report.mismatched.is_empty())
        {
            return Err(Error::State(format!("strict load failed: {report:?}")));
        }

        Ok(report)
    }
}

fn entry_to_blob<B, D>(e: &Entry<B, D>) -> Result<TensorBlob<D>>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
{
    let t = e.tensor.lock().unwrap();
    Ok(TensorBlob {
        kind: e.kind,
        group: e.group.load(std::sync::atomic::Ordering::Relaxed),
        shape: e.shape.clone(),
        data: t.to_vec()?,
    })
}
