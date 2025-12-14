use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use bolt_core::BaseBackend;
use bolt_core::Tensor;

use crate::Float;

static NEXT_PARAM_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ParamId(u64);

impl ParamId {
    pub(crate) fn next() -> Self {
        ParamId(NEXT_PARAM_ID.fetch_add(1, Ordering::Relaxed))
    }
}

pub trait HasParams<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn visit_params<'a>(&'a self, f: &mut dyn FnMut(&'a Parameter<B, D>));
    fn visit_params_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut Parameter<B, D>));

    fn param_count(&self) -> usize {
        let mut count = 0;
        self.visit_params(&mut |_| count += 1);
        count
    }

    fn params(&self) -> Vec<&Parameter<B, D>> {
        let mut params = Vec::with_capacity(self.param_count());
        self.visit_params(&mut |p| params.push(p));
        params
    }

    fn params_mut(&mut self) -> Vec<&mut Parameter<B, D>> {
        let mut params = Vec::with_capacity(self.param_count());
        self.visit_params_mut(&mut |p| params.push(p));
        params
    }

    fn freeze(&mut self) {
        self.visit_params_mut(&mut |p| p.freeze());
    }

    fn unfreeze(&mut self) {
        self.visit_params_mut(&mut |p| p.unfreeze());
    }

    fn zero_grad(&mut self) {
        self.visit_params_mut(&mut |p| p.zero_grad());
    }
}

pub struct Parameter<B, D>
where
    B: BaseBackend,
    D: Float,
{
    data: Tensor<B, D>,
    grad: Option<Tensor<B, D>>,
    id: ParamId,
    name: Option<String>,
    requires_grad: bool,
}

impl<B, D> Parameter<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new(data: Tensor<B, D>) -> Self {
        Self::with_name_opt(data, None)
    }

    pub fn with_name(data: Tensor<B, D>, name: impl Into<String>) -> Self {
        Self::with_name_opt(data, Some(name.into()))
    }

    pub fn with_name_opt(data: Tensor<B, D>, name: Option<String>) -> Self {
        let id = ParamId::next();
        Self {
            data,
            grad: None,
            id,
            name,
            requires_grad: true,
        }
    }

    pub fn tensor(&self) -> &Tensor<B, D> {
        &self.data
    }

    pub fn tensor_mut(&mut self) -> &mut Tensor<B, D> {
        &mut self.data
    }

    pub fn grad(&self) -> Option<&Tensor<B, D>> {
        self.grad.as_ref()
    }

    pub fn grad_mut(&mut self) -> Option<&mut Tensor<B, D>> {
        self.grad.as_mut()
    }

    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub fn id(&self) -> ParamId {
        self.id
    }

    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub fn display_name(&self) -> String {
        match &self.name {
            Some(name) => name.clone(),
            None => format!("p{}", self.id.0),
        }
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    pub fn freeze(&mut self) {
        self.requires_grad = false;
    }

    pub fn unfreeze(&mut self) {
        self.requires_grad = true;
    }

    pub fn set_grad(&mut self, grad: Tensor<B, D>) {
        self.grad = Some(grad);
    }
}
