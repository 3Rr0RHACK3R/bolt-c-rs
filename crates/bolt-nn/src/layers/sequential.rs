use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{Module, Result};

pub struct Seq<B, D>
where
    B: BaseBackend,
    D: Float,
{
    layers: Vec<Box<dyn Module<B, D>>>,
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
    fn forward(&self, mut x: Tensor<B, D>, train: bool) -> Result<Tensor<B, D>> {
        for layer in &self.layers {
            x = layer.forward(x, train)?;
        }
        Ok(x)
    }
}
