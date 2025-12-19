use bolt_autodiff::Float;
use bolt_core::backend::{AddOp, MatmulOp, ReluOp, ReshapeOp, TransposeOp};
use bolt_core::{Backend, BaseBackend};

/// Supertrait alias for required ops on the base backend `B`.
pub trait BaseMlpOps<D: Float>: BaseBackend + MatmulOp<D> + AddOp<D> + TransposeOp<D> {}

impl<T, D> BaseMlpOps<D> for T
where
    T: BaseBackend + MatmulOp<D> + AddOp<D> + TransposeOp<D>,
    D: Float,
{
}

/// Supertrait alias for required ops on the mode backend `M::Backend`.
pub trait TensorMlpOps<D: Float>:
    Backend + ReshapeOp<D> + MatmulOp<D> + AddOp<D> + TransposeOp<D> + ReluOp<D>
{
}

impl<T, D> TensorMlpOps<D> for T
where
    T: Backend + ReshapeOp<D> + MatmulOp<D> + AddOp<D> + TransposeOp<D> + ReluOp<D>,
    D: Float,
{
}
