use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::Arc;

use bolt_autodiff::{Autodiff, Float, Parameter};
use bolt_core::backend::{AddOp, CopyOp, FillOp, SumOp};
use bolt_core::{BaseBackend, Tensor};

use crate::error::Result;
use crate::mode::{Eval, Grad, Mode};

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct Context<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    backend: Arc<B>,
    mode: M,
    rng: RefCell<Rng>,
    _dtype: PhantomData<D>,
}

// ============ Eval Context ============

impl<B, D> Context<B, D, Eval<B, D>>
where
    B: BaseBackend,
    D: Float,
{
    pub fn eval(backend: &Arc<B>) -> Self {
        Self {
            backend: backend.clone(),
            mode: Eval::new(),
            rng: RefCell::new(Rng::new(0)),
            _dtype: PhantomData,
        }
    }
}

// ============ Grad Context ============

impl<B, D> Context<B, D, Grad<B, D>>
where
    B: BaseBackend,
    D: Float,
{
    pub fn grad(backend: &Arc<B>) -> Self {
        let autodiff = Arc::new(Autodiff::wrap(backend.clone()));

        Self {
            backend: backend.clone(),
            mode: Grad::new(autodiff),
            rng: RefCell::new(Rng::new(42)),
            _dtype: PhantomData,
        }
    }

    pub fn autodiff(&self) -> &Arc<Autodiff<B, D>> {
        self.mode.autodiff()
    }

    pub fn backward(
        &self,
        loss: &Tensor<Autodiff<B, D>, D>,
        params: &mut [&mut Parameter<B, D>],
    ) -> Result<()>
    where
        B: AddOp<D> + FillOp<D> + CopyOp<D> + SumOp<D>,
    {
        self.mode.backward(loss, params)
    }
}

// ============ Common Methods ============

impl<B, D, M> Context<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    pub fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    pub fn device(&self) -> &B::Device {
        self.backend.device()
    }

    pub fn rng(&self) -> std::cell::RefMut<'_, Rng> {
        self.rng.borrow_mut()
    }

    pub fn input(&self, tensor: &Tensor<B, D>) -> Tensor<M::Backend, D> {
        self.mode.wrap_input(tensor)
    }

    pub fn param(&self, p: &Parameter<B, D>) -> Tensor<M::Backend, D> {
        self.mode.wrap_param(p)
    }

    pub fn param_frozen(&self, p: &Parameter<B, D>) -> Tensor<M::Backend, D> {
        self.mode.wrap_param_frozen(p)
    }
}
