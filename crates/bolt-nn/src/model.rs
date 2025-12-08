use bolt_core::Backend;
use bolt_core::dtype::NativeType;

use crate::Context;

pub trait Model<B, D>: Send + Sync
where
    B: Backend,
    D: NativeType,
{
    type Input;
    type Output;

    fn forward(&self, ctx: &Context<B>, input: Self::Input) -> Self::Output;
}
