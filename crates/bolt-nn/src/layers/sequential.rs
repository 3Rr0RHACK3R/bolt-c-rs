use bolt_core::Backend;
use bolt_core::Tensor;
use bolt_core::dtype::FloatType;

use crate::context::Context;
use crate::error::Result;
use crate::model::Model;

pub struct Then<A, B> {
    first: A,
    second: B,
}

impl<A, B> Then<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<Bk, D, A, B> Model<Bk, D> for Then<A, B>
where
    Bk: Backend,
    D: FloatType,
    A: Model<Bk, D, Input = Tensor<Bk, D>, Output = Result<Tensor<Bk, D>>>,
    B: Model<Bk, D, Input = Tensor<Bk, D>, Output = Result<Tensor<Bk, D>>>,
{
    type Input = Tensor<Bk, D>;
    type Output = Result<Tensor<Bk, D>>;

    fn forward(&self, ctx: &Context<Bk>, input: Self::Input) -> Self::Output {
        let mid = self.first.forward(ctx, input)?;
        self.second.forward(ctx, mid)
    }
}

pub trait ModelExt<B, D>: Model<B, D> + Sized
where
    B: Backend,
    D: FloatType,
{
    fn then<N>(self, next: N) -> Then<Self, N>
    where
        N: Model<B, D>,
    {
        Then::new(self, next)
    }
}

impl<T, B, D> ModelExt<B, D> for T
where
    T: Model<B, D>,
    B: Backend,
    D: FloatType,
{
}

pub struct Seq<B, D>
where
    B: Backend,
    D: bolt_core::dtype::NativeType,
{
    layers: Vec<Box<dyn Model<B, D, Input = Tensor<B, D>, Output = Result<Tensor<B, D>>> + Send + Sync>>,
}

impl<B, D> Seq<B, D>
where
    B: Backend,
    D: FloatType,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn push<M>(mut self, layer: M) -> Self
    where
        M: Model<B, D, Input = Tensor<B, D>, Output = Result<Tensor<B, D>>> + Send + Sync + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

impl<B, D> Default for Seq<B, D>
where
    B: Backend,
    D: FloatType,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D> Model<B, D> for Seq<B, D>
where
    B: Backend,
    D: FloatType,
{
    type Input = Tensor<B, D>;
    type Output = Result<Tensor<B, D>>;

    fn forward(&self, ctx: &Context<B>, input: Self::Input) -> Self::Output {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(ctx, x)?;
        }
        Ok(x)
    }
}
