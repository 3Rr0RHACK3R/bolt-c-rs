use bolt_autodiff::Float;
use bolt_core::BaseBackend;

use crate::{Context, Mode};

pub trait Model<B, D, M>: Send + Sync
where
    B: BaseBackend,
    D: Float,
    M: Mode<B, D>,
{
    type Input;
    type Output;

    fn forward(&self, ctx: &Context<B, D, M>, input: Self::Input) -> Self::Output;
}
