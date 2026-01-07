use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

pub struct Seq<B, D>
where
    B: BaseBackend,
    D: Float,
{
    layers: Vec<Box<dyn Module<B, D>>>,
}

impl<B, D> Default for Seq<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D> Seq<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn push<M>(mut self, layer: M) -> Self
    where
        M: Module<B, D> + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

impl<B, D> Module<B, D> for Seq<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn forward(&self, mut x: Tensor<B, D>, ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        for layer in &self.layers {
            x = layer.forward(x, ctx)?;
        }
        Ok(x)
    }
}
