use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use bolt_autodiff::{Autodiff, AutodiffStorage, Float, GradContext, Handle, ParamId, Parameter};
use bolt_core::backend::{AddOp, CopyOp, FillOp, SumOp};
use bolt_core::{BaseBackend, OneValue, Tensor};
use tinyvec::ArrayVec;

use crate::error::{Error, Result};

pub trait Mode<B: BaseBackend, D: Float>: Sized {
    type Backend: bolt_core::Backend;

    fn wrap_input(&self, tensor: &Tensor<B, D>) -> Tensor<Self::Backend, D>;
    fn wrap_param(&self, p: &Parameter<B, D>) -> Tensor<Self::Backend, D>;
    fn wrap_param_frozen(&self, p: &Parameter<B, D>) -> Tensor<Self::Backend, D>;
}

pub struct Eval<B, D> {
    _marker: PhantomData<(B, D)>,
}

impl<B, D> Eval<B, D> {
    pub(crate) fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<B, D> Mode<B, D> for Eval<B, D>
where
    B: BaseBackend,
    D: Float,
{
    type Backend = B;

    fn wrap_input(&self, tensor: &Tensor<B, D>) -> Tensor<B, D> {
        tensor.clone()
    }

    fn wrap_param(&self, p: &Parameter<B, D>) -> Tensor<B, D> {
        p.tensor().clone()
    }

    fn wrap_param_frozen(&self, p: &Parameter<B, D>) -> Tensor<B, D> {
        p.tensor().clone()
    }
}

pub struct Grad<B, D>
where
    B: BaseBackend,
    D: Float,
{
    autodiff: Arc<Autodiff<B, D>>,
    grad_ctx: RefCell<Option<GradContext<B, D>>>,
    param_handles: RefCell<HashMap<ParamId, Handle>>,
}

impl<B, D> Grad<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub(crate) fn new(autodiff: Arc<Autodiff<B, D>>) -> Self {
        // Start the gradient graph immediately
        let grad_ctx = autodiff.begin_grad();

        Self {
            autodiff,
            grad_ctx: RefCell::new(Some(grad_ctx)),
            param_handles: RefCell::new(HashMap::new()),
        }
    }

    pub fn autodiff(&self) -> &Arc<Autodiff<B, D>> {
        &self.autodiff
    }

    pub fn backward(
        &self,
        loss: &Tensor<Autodiff<B, D>, D>,
        params: &mut [&mut Parameter<B, D>],
    ) -> Result<()>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
        D: OneValue,
    {
        let grad_ctx_opt = self.grad_ctx.borrow();
        let grad_ctx = grad_ctx_opt.as_ref().ok_or_else(|| {
            Error::MissingParam(
                "No active gradient context. backward() may have already been called.".into(),
            )
        })?;

        let grads = grad_ctx.backward(loss)?;

        let param_handles = self.param_handles.borrow();
        for param in params.iter_mut() {
            if let Some(&handle) = param_handles.get(&param.id()) {
                if let Some(grad) = grads.get(&handle) {
                    param.set_grad(grad.clone());
                } else {
                    param.zero_grad();
                }
            }
        }

        Ok(())
    }
}

impl<B, D> Mode<B, D> for Grad<B, D>
where
    B: BaseBackend,
    D: Float,
{
    type Backend = Autodiff<B, D>;

    fn wrap_input(&self, tensor: &Tensor<B, D>) -> Tensor<Autodiff<B, D>, D> {
        let layout = tensor.layout().clone();
        let storage = AutodiffStorage::new(tensor.storage().clone(), Handle::NONE, false);
        Tensor::from_parts(self.autodiff.clone(), storage, layout)
    }

    fn wrap_param(&self, p: &Parameter<B, D>) -> Tensor<Autodiff<B, D>, D> {
        let layout = p.tensor().layout().clone();

        // If not tracking gradients for this param, return frozen
        if !p.requires_grad() {
            return self.wrap_param_frozen(p);
        }

        // Check if already tracked (parameter sharing)
        if let Some(&handle) = self.param_handles.borrow().get(&p.id()) {
            let storage = AutodiffStorage::new(p.tensor().storage().clone(), handle, true);
            return Tensor::from_parts(self.autodiff.clone(), storage, layout);
        }

        // Create tracked storage via autodiff graph
        let storage = self.autodiff.create_tracked_storage(
            p.tensor().storage().clone(),
            &layout,
            true, // requires_grad
            true, // is_leaf
            ArrayVec::new(),
            None,
            vec![],
        );

        let handle = storage.handle();
        self.param_handles.borrow_mut().insert(p.id(), handle);

        Tensor::from_parts(self.autodiff.clone(), storage, layout)
    }

    fn wrap_param_frozen(&self, p: &Parameter<B, D>) -> Tensor<Autodiff<B, D>, D> {
        self.wrap_input(p.tensor())
    }
}
