use bolt_autodiff::{Float, HasParams, Parameter};
use bolt_core::Backend;
use bolt_core::BaseBackend;
use bolt_core::Tensor;

use crate::context::Context;
use crate::error::Result;
use crate::mode::{Eval, Grad, Mode};
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

impl<Bk, D, A, B> HasParams<Bk, D> for Then<A, B>
where
    Bk: BaseBackend,
    D: Float,
    A: HasParams<Bk, D>,
    B: HasParams<Bk, D>,
{
    fn params(&self) -> Vec<&Parameter<Bk, D>> {
        let mut params = self.first.params();
        params.extend(self.second.params());
        params
    }

    fn params_mut(&mut self) -> Vec<&mut Parameter<Bk, D>> {
        let mut params = self.first.params_mut();
        params.extend(self.second.params_mut());
        params
    }
}

impl<Bk, D, M, A, B> Model<Bk, D, M> for Then<A, B>
where
    Bk: BaseBackend,
    D: Float,
    M: Mode<Bk, D>,
    M::Backend: Backend,
    A: Model<Bk, D, M, Input = Tensor<M::Backend, D>, Output = Result<Tensor<M::Backend, D>>>,
    B: Model<Bk, D, M, Input = Tensor<M::Backend, D>, Output = Result<Tensor<M::Backend, D>>>,
{
    type Input = Tensor<M::Backend, D>;
    type Output = Result<Tensor<M::Backend, D>>;

    fn forward(&self, ctx: &Context<Bk, D, M>, input: Self::Input) -> Self::Output {
        let mid = self.first.forward(ctx, input)?;
        self.second.forward(ctx, mid)
    }
}

pub trait ModelExt<B, D>: Sized
where
    B: BaseBackend,
    D: Float,
{
    fn then<N>(self, next: N) -> Then<Self, N>
    where
        Self: Model<B, D, Eval<B, D>> + Model<B, D, Grad<B, D>>,
        N: Model<B, D, Eval<B, D>> + Model<B, D, Grad<B, D>>,
    {
        Then::new(self, next)
    }
}

impl<T, B, D> ModelExt<B, D> for T
where
    T: Model<B, D, Eval<B, D>> + Model<B, D, Grad<B, D>>,
    B: BaseBackend,
    D: Float,
{
}

pub trait Layer<B, D, M>:
    Model<B, D, M, Input = Tensor<M::Backend, D>, Output = Result<Tensor<M::Backend, D>>>
    + HasParams<B, D>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
}

impl<T, B, D, M> Layer<B, D, M> for T
where
    T: Model<B, D, M, Input = Tensor<M::Backend, D>, Output = Result<Tensor<M::Backend, D>>>
        + HasParams<B, D>,
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
    M::Backend: Backend,
{
}

pub struct Seq<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    layers: Vec<Box<dyn Layer<B, D, M> + Send + Sync>>,
}

impl<B, D, M> Seq<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn push<L>(mut self, layer: L) -> Self
    where
        L: Layer<B, D, M> + Send + Sync + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }
}

impl<B, D, M> Default for Seq<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<B, D, M> Model<B, D, M> for Seq<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
    M::Backend: Backend,
{
    type Input = Tensor<M::Backend, D>;
    type Output = Result<Tensor<M::Backend, D>>;

    fn forward(&self, ctx: &Context<B, D, M>, input: Self::Input) -> Self::Output {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(ctx, x)?;
        }
        Ok(x)
    }
}

impl<B, D, M> HasParams<B, D> for Seq<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    fn params(&self) -> Vec<&Parameter<B, D>> {
        let mut all = Vec::new();
        for layer in &self.layers {
            all.extend(layer.params());
        }
        all
    }

    fn params_mut(&mut self) -> Vec<&mut Parameter<B, D>> {
        let mut all = Vec::new();
        for layer in &mut self.layers {
            all.extend(layer.params_mut());
        }
        all
    }
}
