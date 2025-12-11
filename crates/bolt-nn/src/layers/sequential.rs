use bolt_autodiff::Float;
use bolt_core::Backend;
use bolt_core::BaseBackend;
use bolt_core::Tensor;

use crate::context::Context;
use crate::error::Result;
use crate::mode::Mode;
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

pub trait ModelExt<B, D, M>: Model<B, D, M> + Sized
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    fn then<N>(self, next: N) -> Then<Self, N>
    where
        N: Model<B, D, M>,
    {
        Then::new(self, next)
    }
}

impl<T, B, D, M> ModelExt<B, D, M> for T
where
    T: Model<B, D, M>,
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
}

pub struct Seq<B, D, M>
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    layers: Vec<
        Box<
            dyn Model<
                    B,
                    D,
                    M,
                    Input = Tensor<M::Backend, D>,
                    Output = Result<Tensor<M::Backend, D>>,
                > + Send
                + Sync,
        >,
    >,
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
        L: Model<B, D, M, Input = Tensor<M::Backend, D>, Output = Result<Tensor<M::Backend, D>>>
            + Send
            + Sync
            + 'static,
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
