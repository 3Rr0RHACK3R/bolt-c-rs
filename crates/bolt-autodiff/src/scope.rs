use std::marker::PhantomData;
use std::sync::{Arc, RwLock};

use bolt_core::backend::{AddOp, CopyOp, FillOp, SumOp};
use bolt_core::{BaseBackend, Float, Tensor};

use crate::error::Result;
use crate::grad_tape::GradTape;
use crate::gradients::Gradients;
use crate::operations::Autodiff;
use crate::tensor_ext::AutodiffTensorExt;

pub struct GradContext<B, D>
where
    B: BaseBackend,
    D: Float,
{
    autodiff: Autodiff<B, D>,
}

impl<B, D> GradContext<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub(crate) fn new(autodiff: Autodiff<B, D>) -> Self {
        Self { autodiff }
    }

    pub fn autodiff(&self) -> &Autodiff<B, D> {
        &self.autodiff
    }

    pub fn backward(&self, loss: &Tensor<Autodiff<B, D>, D>) -> Result<Gradients<B, D>>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    {
        loss.backward()
    }

    pub fn tape(&self) -> GradTape<'_, B, D> {
        GradTape::new(self)
    }
}

impl<B, D> Drop for GradContext<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn drop(&mut self) {
        *self.autodiff.graph().write().unwrap() = None;
    }
}

pub struct NoGradGuard<B, D>
where
    B: BaseBackend,
    D: Float,
{
    autodiff_grad_enabled: Arc<RwLock<bool>>,
    prev: bool,
    _marker: PhantomData<(B, D)>,
}

impl<B, D> NoGradGuard<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(autodiff: &Autodiff<B, D>) -> Self {
        let lock = autodiff.grad_enabled_lock().clone();
        let prev = *lock.read().unwrap();
        *lock.write().unwrap() = false;
        Self {
            autodiff_grad_enabled: lock,
            prev,
            _marker: PhantomData,
        }
    }
}

impl<B, D> Drop for NoGradGuard<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn drop(&mut self) {
        *self.autodiff_grad_enabled.write().unwrap() = self.prev;
    }
}
