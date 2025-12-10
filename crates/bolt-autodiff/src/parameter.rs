use std::sync::atomic::{AtomicU64, Ordering};

use bolt_core::backend::Backend;
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

pub struct Parameter<B, D>
where
    B: Backend,
    D: Float,
{
    value: Tensor<B, D>,
    grad: Option<Tensor<B, D>>,
    id: ParamId,
    name: Option<String>,
}

impl<B, D> Parameter<B, D>
where
    B: Backend,
    D: Float,
{
    pub fn new(value: Tensor<B, D>) -> Self {
        Self::with_name_opt(value, None)
    }

    pub fn with_name(value: Tensor<B, D>, name: impl Into<String>) -> Self {
        Self::with_name_opt(value, Some(name.into()))
    }

    pub fn with_name_opt(value: Tensor<B, D>, name: Option<String>) -> Self {
        let id = ParamId::next();
        Self {
            value,
            grad: None,
            id,
            name,
        }
    }

    pub fn value(&self) -> &Tensor<B, D> {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut Tensor<B, D> {
        &mut self.value
    }

    pub fn grad(&self) -> Option<&Tensor<B, D>> {
        self.grad.as_ref()
    }

    pub fn grad_mut(&mut self) -> Option<&mut Tensor<B, D>> {
        self.grad.as_mut()
    }

    pub fn clear_grad(&mut self) {
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

    pub(crate) fn set_grad(&mut self, grad: Tensor<B, D>) {
        self.grad = Some(grad);
    }
}
