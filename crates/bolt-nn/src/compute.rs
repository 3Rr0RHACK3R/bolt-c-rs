use bolt_autodiff::Float;
use bolt_core::backend::{
    AddOp, Backend, CopyOp, FillOp, MatmulOp, ReshapeOp, ReluOp, SumOp, TransposeOp,
};
use bolt_core::BaseBackend;

pub trait Compute<D: Float>:
    BaseBackend
    + Backend
    + AddOp<D>
    + CopyOp<D>
    + FillOp<D>
    + MatmulOp<D>
    + ReshapeOp<D>
    + ReluOp<D>
    + SumOp<D>
    + TransposeOp<D>
{
}

impl<B, D> Compute<D> for B
where
    B: BaseBackend
        + Backend
        + AddOp<D>
        + CopyOp<D>
        + FillOp<D>
        + MatmulOp<D>
        + ReshapeOp<D>
        + ReluOp<D>
        + SumOp<D>
        + TransposeOp<D>,
    D: Float,
{
}

pub trait ComputeOps<D: Float>:
    Backend
    + AddOp<D>
    + CopyOp<D>
    + FillOp<D>
    + MatmulOp<D>
    + ReshapeOp<D>
    + ReluOp<D>
    + SumOp<D>
    + TransposeOp<D>
{
}

impl<B, D> ComputeOps<D> for B
where
    B: Backend
        + AddOp<D>
        + CopyOp<D>
        + FillOp<D>
        + MatmulOp<D>
        + ReshapeOp<D>
        + ReluOp<D>
        + SumOp<D>
        + TransposeOp<D>,
    D: Float,
{
}
