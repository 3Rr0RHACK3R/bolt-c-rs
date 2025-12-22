use bolt_core::Backend;
use bolt_core::backend::{SqueezeOp, UnsqueezeOp};
use bolt_core::dtype::NativeType;
use bolt_core::error::Result;

use crate::Tensor;
use crate::autograd::{BackwardContext, BackwardOp};

pub(crate) struct SqueezeAxisBackward {
    axis: isize,
}

impl SqueezeAxisBackward {
    pub(crate) fn new(axis: isize) -> Self {
        Self { axis }
    }
}

impl<B, D> BackwardOp<B, D> for SqueezeAxisBackward
where
    B: Backend + SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        Ok(vec![Some(grad_output.unsqueeze(self.axis)?)])
    }

    fn name(&self) -> &'static str {
        "SqueezeAxisBackward"
    }
}

pub(crate) struct SqueezeAllBackward {
    axes: Vec<usize>,
}

impl SqueezeAllBackward {
    pub(crate) fn new(axes: Vec<usize>) -> Self {
        Self { axes }
    }
}

impl<B, D> BackwardOp<B, D> for SqueezeAllBackward
where
    B: Backend + SqueezeOp<D> + UnsqueezeOp<D> + 'static,
    D: NativeType + 'static,
{
    fn backward(
        &self,
        grad_output: &Tensor<B, D>,
        _ctx: &BackwardContext<'_, B, D>,
    ) -> Result<Vec<Option<Tensor<B, D>>>> {
        let mut g = grad_output.clone();
        for &axis in &self.axes {
            g = g.unsqueeze(axis as isize)?;
        }
        Ok(vec![Some(g)])
    }

    fn name(&self) -> &'static str {
        "SqueezeAllBackward"
    }
}
