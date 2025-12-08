use std::sync::Arc;

use bolt_core::Backend;
use bolt_core::Tensor;
use tinyvec::ArrayVec;

use crate::Float;
use crate::error::Result;

pub const MAX_INPUTS: usize = 8;

pub struct BackwardContext<'a, B, D>
where
    B: Backend<D>,
    D: Float,
{
    saved: &'a [Tensor<B, D>],
    backend: &'a Arc<B>,
}

impl<'a, B, D> BackwardContext<'a, B, D>
where
    B: Backend<D>,
    D: Float,
{
    pub fn new(saved: &'a [Tensor<B, D>], backend: &'a Arc<B>) -> Self {
        Self { saved, backend }
    }

    pub fn saved(&self, idx: usize) -> &Tensor<B, D> {
        &self.saved[idx]
    }

    pub fn num_saved(&self) -> usize {
        self.saved.len()
    }

    pub fn backend(&self) -> &Arc<B> {
        self.backend
    }
}

pub trait BackwardOp<B, D>: Send + Sync
where
    B: Backend<D>,
    D: Float,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        ctx: &BackwardContext<B, D>,
    ) -> Result<ArrayVec<[Option<Tensor<B, D>>; MAX_INPUTS]>>;

    fn name(&self) -> &'static str;
}
