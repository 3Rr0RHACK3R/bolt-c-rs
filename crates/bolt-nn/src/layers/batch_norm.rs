use bolt_core::backend::{
    AddOp, BroadcastToOp, CopyOp, DivOp, FillOp, MeanOp, MulOp, NegOp, ReshapeOp, SqrtOp,
    SqueezeOp, SubOp, SumOp, UnsqueezeOp,
};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{Error, ForwardCtx, Module, Result, Store};

use super::norm::{Norm, NormConfig};

pub struct BatchNorm<B, D>
where
    B: BaseBackend,
    D: Float,
{
    inner: Norm<B, D>,
    num_features: usize,
}

impl<B, D> BatchNorm<B, D>
where
    B: BaseBackend,
    D: Float,
{
    pub fn init(
        store: &Store<B, D>,
        num_features: usize,
        affine: bool,
        eps: f64,
        momentum: f64,
    ) -> Result<Self> {
        // BatchNorm uses Norm for param/buffer storage only; axes are computed
        // dynamically in forward() based on input rank, so axes is None.
        let config = NormConfig {
            axes: None,
            normalized_shape: vec![num_features],
            affine,
            track_running_stats: true,
            eps,
            momentum,
        };

        let inner = Norm::init(store, config)?;

        Ok(Self {
            inner,
            num_features,
        })
    }

    pub fn init_default(store: &Store<B, D>, num_features: usize) -> Result<Self> {
        Self::init(store, num_features, true, 1e-5, 0.1)
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }

    pub fn eps(&self) -> f64 {
        self.inner.eps()
    }

    pub fn momentum(&self) -> f64 {
        self.inner.momentum()
    }

    fn reduction_axes(&self, rank: usize) -> Result<Vec<isize>> {
        match rank {
            2 => Ok(vec![0]),
            3 => Ok(vec![0, 2]),
            4 => Ok(vec![0, 2, 3]),
            5 => Ok(vec![0, 2, 3, 4]),
            _ => Err(Error::Shape(format!(
                "BatchNorm: unsupported input rank {}, expected 2-5",
                rank
            ))),
        }
    }
}

impl<B, D> Module<B, D> for BatchNorm<B, D>
where
    B: BaseBackend
        + AddOp<D>
        + BroadcastToOp<D>
        + CopyOp<D>
        + DivOp<D>
        + FillOp<D>
        + MeanOp<D>
        + MulOp<D>
        + NegOp<D>
        + ReshapeOp<D>
        + SqueezeOp<D>
        + SqrtOp<D>
        + SubOp<D>
        + SumOp<D>
        + UnsqueezeOp<D>
        + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        let shape = x.shape();
        let rank = shape.len();

        if rank < 2 {
            return Err(Error::Shape(format!(
                "BatchNorm: input must be at least 2D, got {}D",
                rank
            )));
        }

        if shape[1] != self.num_features {
            return Err(Error::Shape(format!(
                "BatchNorm: expected {} features at dim 1, got {}",
                self.num_features, shape[1]
            )));
        }

        let reduce_axes = self.reduction_axes(rank)?;
        let param_axes = vec![1isize];

        self.inner.forward(x, ctx, &reduce_axes, &param_axes)
    }
}
