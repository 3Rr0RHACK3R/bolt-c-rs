use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use bolt_core::BaseBackend;
use bolt_core::Float;
use bolt_core::backend::{AddOp, FillOp};
use bolt_rng::RngStream;
use bolt_tensor::Tensor;

use crate::init;
use crate::{Error, Init, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Param,
    Buffer,
}

pub(crate) struct Entry<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub(crate) key: String,
    pub(crate) kind: Kind,
    pub(crate) group: AtomicU32,
    pub(crate) shape: Vec<usize>,
    pub(crate) requires_grad: AtomicBool,
    pub(crate) tensor: Mutex<Tensor<B, D>>,
    pub(crate) grad: Mutex<Option<Tensor<B, D>>>,
}

#[derive(Clone)]
pub struct Param<B, D>(Arc<Entry<B, D>>)
where
    B: BaseBackend,
    D: Float;

#[derive(Clone)]
pub struct Buffer<B, D>(Arc<Entry<B, D>>)
where
    B: BaseBackend,
    D: Float;

impl<B, D> Param<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn key(&self) -> &str {
        &self.0.key
    }

    pub fn group(&self) -> u32 {
        self.0.group.load(Ordering::Relaxed)
    }

    pub fn set_group(&self, group: u32) {
        self.0.group.store(group, Ordering::Relaxed);
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad.load(Ordering::Relaxed)
    }

    pub fn freeze(&self) {
        self.set_requires_grad(false);
    }

    pub fn unfreeze(&self) {
        self.set_requires_grad(true);
    }

    pub fn set_requires_grad(&self, on: bool) {
        self.0.requires_grad.store(on, Ordering::Relaxed);
        self.0.grad.lock().unwrap().take();
        let mut t = self.0.tensor.lock().unwrap();
        *t = t.clone().with_requires_grad(on);
    }

    pub fn tensor(&self) -> Tensor<B, D> {
        self.0.tensor.lock().unwrap().clone()
    }

    pub fn set_tensor(&self, tensor: Tensor<B, D>) -> Result<()> {
        if tensor.shape() != self.0.shape.as_slice() {
            return Err(Error::Shape(format!(
                "param {}: expected shape {:?}, got {:?}",
                self.key(),
                self.0.shape,
                tensor.shape()
            )));
        }
        let on = self.requires_grad();
        *self.0.tensor.lock().unwrap() = tensor.with_requires_grad(on);
        Ok(())
    }

    pub fn grad(&self) -> Option<Tensor<B, D>> {
        self.0.grad.lock().unwrap().clone()
    }

    pub fn set_grad(&self, grad: Option<Tensor<B, D>>) {
        *self.0.grad.lock().unwrap() = grad;
    }

    pub fn zero_grad(&self) {
        self.0.grad.lock().unwrap().take();
    }
}

impl<B, D> Buffer<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn key(&self) -> &str {
        &self.0.key
    }

    pub fn shape(&self) -> &[usize] {
        &self.0.shape
    }

    pub fn tensor(&self) -> Tensor<B, D> {
        self.0.tensor.lock().unwrap().clone()
    }

    pub fn set(&self, tensor: Tensor<B, D>) -> Result<()> {
        if tensor.shape() != self.0.shape.as_slice() {
            return Err(Error::Shape(format!(
                "buffer {}: expected shape {:?}, got {:?}",
                self.key(),
                self.0.shape,
                tensor.shape()
            )));
        }
        *self.0.tensor.lock().unwrap() = tensor.with_requires_grad(false);
        Ok(())
    }
}

struct Inner<B, D>
where
    B: BaseBackend,
    D: Float,
{
    backend: Arc<B>,
    params: RwLock<BTreeMap<String, Arc<Entry<B, D>>>>,
    buffers: RwLock<BTreeMap<String, Arc<Entry<B, D>>>>,
    sealed: AtomicBool,
    rng: Mutex<RngStream>,
}

#[derive(Clone)]
pub struct Store<B, D>
where
    B: BaseBackend,
    D: Float,
{
    inner: Arc<Inner<B, D>>,
    prefix: String,
    group: u32,
}

impl<B, D> Store<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(backend: Arc<B>, seed: u64) -> Self {
        Self {
            inner: Arc::new(Inner {
                backend,
                params: RwLock::new(BTreeMap::new()),
                buffers: RwLock::new(BTreeMap::new()),
                sealed: AtomicBool::new(false),
                rng: Mutex::new(RngStream::from_seed(seed)),
            }),
            prefix: String::new(),
            group: 0,
        }
    }

    pub fn backend(&self) -> Arc<B> {
        self.inner.backend.clone()
    }

    pub fn sub(&self, name: &str) -> Self {
        let mut prefix = self.prefix.clone();
        if !prefix.is_empty() {
            prefix.push('.');
        }
        prefix.push_str(name);
        Self {
            inner: self.inner.clone(),
            prefix,
            group: self.group,
        }
    }

    pub fn sub_idx(&self, idx: usize) -> Self {
        self.sub(&idx.to_string())
    }

    pub fn group(&self, group: u32) -> Self {
        Self {
            inner: self.inner.clone(),
            prefix: self.prefix.clone(),
            group,
        }
    }

    pub fn seal(&self) {
        self.inner.sealed.store(true, Ordering::Relaxed);
    }

    pub fn param(&self, name: &str, shape: &[usize], initv: Init<D>) -> Result<Param<B, D>> {
        self.create(name, shape, initv, Kind::Param).map(Param)
    }

    pub fn buffer(&self, name: &str, shape: &[usize], initv: Init<D>) -> Result<Buffer<B, D>> {
        self.create(name, shape, initv, Kind::Buffer).map(Buffer)
    }

    pub fn trainable(&self) -> Vec<Param<B, D>> {
        let map = self.inner.params.read().unwrap();
        map.values().cloned().map(Param).collect()
    }

    pub fn named_trainable(&self) -> Vec<(String, Param<B, D>)> {
        let map = self.inner.params.read().unwrap();
        map.iter()
            .map(|(k, v)| (k.clone(), Param(v.clone())))
            .collect()
    }

    pub fn trainable_by_group(&self, group_id: u32) -> Vec<Param<B, D>> {
        let map = self.inner.params.read().unwrap();
        map.values()
            .filter(|e| e.group.load(Ordering::Relaxed) == group_id)
            .cloned()
            .map(Param)
            .collect()
    }

    pub fn group_params_by_name<F>(&self, predicate: F, group_id: u32)
    where
        F: Fn(&str) -> bool,
    {
        let map = self.inner.params.read().unwrap();
        for entry in map.values() {
            if predicate(&entry.key) {
                entry.group.store(group_id, Ordering::Relaxed);
            }
        }
    }

    pub fn zero_grad(&self) {
        for p in self.trainable() {
            p.zero_grad();
        }
    }

    pub fn backward(&self, loss: &Tensor<B, D>) -> Result<()>
    where
        B: AddOp<D> + FillOp<D> + 'static,
        D: 'static,
    {
        let grads = loss.backward()?;
        for p in self.trainable() {
            if !p.requires_grad() {
                p.set_grad(None);
                continue;
            }
            let g = grads.wrt(&p.tensor()).cloned();
            p.set_grad(g);
        }
        Ok(())
    }

    fn create(
        &self,
        name: &str,
        shape: &[usize],
        initv: Init<D>,
        kind: Kind,
    ) -> Result<Arc<Entry<B, D>>> {
        if self.inner.sealed.load(Ordering::Relaxed) {
            return Err(Error::State(
                "Store is sealed; cannot create new params/buffers".into(),
            ));
        }

        let key = if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        };

        let mut rng = self.inner.rng.lock().unwrap();
        let data = init::fill(shape, initv, &mut rng)?;

        let tensor = Tensor::<B, D>::from_vec(&self.inner.backend, data, shape)?;
        let requires_grad = kind == Kind::Param;
        let tensor = tensor.with_requires_grad(requires_grad);

        let entry = Arc::new(Entry {
            key: key.clone(),
            kind,
            group: AtomicU32::new(self.group),
            shape: shape.to_vec(),
            requires_grad: AtomicBool::new(requires_grad),
            tensor: Mutex::new(tensor),
            grad: Mutex::new(None),
        });

        match kind {
            Kind::Param => {
                let mut map = self.inner.params.write().unwrap();
                if map.contains_key(&key) {
                    return Err(Error::State(format!("duplicate param key: {key}")));
                }
                map.insert(key, entry.clone());
            }
            Kind::Buffer => {
                let mut map = self.inner.buffers.write().unwrap();
                if map.contains_key(&key) {
                    return Err(Error::State(format!("duplicate buffer key: {key}")));
                }
                map.insert(key, entry.clone());
            }
        }

        Ok(entry)
    }

    pub(crate) fn all_entries(&self) -> (Vec<Arc<Entry<B, D>>>, Vec<Arc<Entry<B, D>>>) {
        let ps = self
            .inner
            .params
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect();
        let bs = self
            .inner
            .buffers
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect();
        (ps, bs)
    }

    pub(crate) fn get_entry(&self, key: &str) -> Option<Arc<Entry<B, D>>> {
        if let Some(e) = self.inner.params.read().unwrap().get(key) {
            return Some(e.clone());
        }
        self.inner.buffers.read().unwrap().get(key).cloned()
    }

    pub(crate) fn expected_keys(&self) -> Vec<String> {
        let mut out: Vec<String> = self.inner.params.read().unwrap().keys().cloned().collect();
        out.extend(self.inner.buffers.read().unwrap().keys().cloned());
        out.sort();
        out
    }
}
