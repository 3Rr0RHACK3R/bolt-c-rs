use bolt_core::backend::{CopyOp, MeanOp};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

pub struct GlobalAvgPool {
    keepdim: bool,
}

impl GlobalAvgPool {
    pub fn new(keepdim: bool) -> Self {
        Self { keepdim }
    }
}

impl<B, D> Module<B, D> for GlobalAvgPool
where
    B: BaseBackend + CopyOp<D> + MeanOp<D> + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        let shape = x.shape();
        let rank = shape.len();
        
        if rank < 3 {
            return Err(crate::Error::Shape(format!(
                "GlobalAvgPool expects at least 3D input (B, C, ...), got {}D",
                rank
            )));
        }
        
        // Average over all spatial dimensions (everything after batch and channels)
        // For rank N, we average over axes [2, 3, ..., N-1]
        let spatial_axes: Vec<isize> = (2..rank as isize).collect();
        Ok(x.mean(Some(&spatial_axes), self.keepdim)?)
    }
}

