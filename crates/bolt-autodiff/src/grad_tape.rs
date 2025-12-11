use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bolt_core::backend::{AddOp, CopyOp, FillOp, SumOp};
use bolt_core::{BaseBackend, OneValue, Tensor};
use tinyvec::ArrayVec;

use crate::error::{Error, Result};
use crate::operations::Autodiff;
use crate::parameter::{ParamId, Parameter};
use crate::scope::GradContext;
use crate::storage::AutodiffStorage;
use crate::{Float, Handle};

pub struct GradTape<'a, B, D>
where
    B: BaseBackend,
    D: Float,
{
    ctx: &'a GradContext<B, D>,
    autodiff: Arc<Autodiff<B, D>>,
    param_handles: HashMap<ParamId, Handle>,
}

impl<'a, B, D> GradTape<'a, B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub(crate) fn new(ctx: &'a GradContext<B, D>) -> Self {
        Self {
            ctx,
            autodiff: Arc::new(ctx.autodiff().clone()),
            param_handles: HashMap::new(),
        }
    }

    pub fn input(&self, base: &Tensor<B, D>) -> Tensor<Autodiff<B, D>, D> {
        let layout = base.layout().clone();
        let storage = AutodiffStorage::new(base.storage().clone(), Handle::NONE, false);
        Tensor::from_parts(self.autodiff.clone(), storage, layout)
    }

    pub fn param(&mut self, p: &Parameter<B, D>) -> Tensor<Autodiff<B, D>, D> {
        let layout = p.tensor().layout().clone();
        let autodiff = self.ctx.autodiff();

        if let Some(handle) = self.param_handles.get(&p.id()) {
            let storage = AutodiffStorage::new(p.tensor().storage().clone(), *handle, true);
            return Tensor::from_parts(self.autodiff.clone(), storage, layout);
        }

        let storage = autodiff.create_tracked_storage(
            p.tensor().storage().clone(),
            &layout,
            p.requires_grad(),
            true,
            ArrayVec::new(),
            None,
            vec![],
        );

        let handle = storage.handle();
        debug_assert_ne!(handle, Handle::NONE, "parameter requires active grad graph");
        self.param_handles.insert(p.id(), handle);

        Tensor::from_parts(self.autodiff.clone(), storage, layout)
    }

    pub fn backward_into_params(
        &mut self,
        loss: &Tensor<Autodiff<B, D>, D>,
        params: &mut [&mut Parameter<B, D>],
    ) -> Result<()>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
        D: OneValue,
    {
        let grads = self.ctx.backward(loss)?;
        let mut seen = HashSet::new();

        for param in params.iter_mut() {
            if !seen.insert(param.id()) {
                continue;
            }

            let handle = *self.param_handles.get(&param.id()).ok_or_else(|| Error::ParamNotInTape {
                param_id: param.id(),
                param_name: param.name().map(|n| n.to_string()),
            })?;

            match grads.get(&handle) {
                Some(grad) => param.set_grad(grad.clone()),
                None => param.zero_grad(),
            }
        }

        Ok(())
    }

    pub fn backward_param_grads(
        &mut self,
        loss: &Tensor<Autodiff<B, D>, D>,
        params: &[&Parameter<B, D>],
    ) -> Result<ParamGrads<B, D>>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
        D: OneValue,
    {
        let grads = self.ctx.backward(loss)?;
        let mut inner = HashMap::new();
        let mut seen = HashSet::new();

        for param in params.iter() {
            if !seen.insert(param.id()) {
                continue;
            }

            let handle = *self.param_handles.get(&param.id()).ok_or_else(|| Error::ParamNotInTape {
                param_id: param.id(),
                param_name: param.name().map(|n| n.to_string()),
            })?;

            if let Some(grad) = grads.get(&handle) {
                inner.insert(param.id(), grad.clone());
            }
        }

        Ok(ParamGrads::new(inner))
    }
}

pub struct ParamGrads<B, D>
where
    B: BaseBackend,
    D: Float,
{
    inner: HashMap<ParamId, Tensor<B, D>>,
}

impl<B, D> ParamGrads<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn get(&self, p: &Parameter<B, D>) -> Option<&Tensor<B, D>> {
        self.inner.get(&p.id())
    }

    pub fn iter(&self) -> impl Iterator<Item = (ParamId, &Tensor<B, D>)> {
        self.inner.iter().map(|(k, v)| (*k, v))
    }

    pub(crate) fn new(inner: HashMap<ParamId, Tensor<B, D>>) -> Self {
        Self { inner }
    }
}
