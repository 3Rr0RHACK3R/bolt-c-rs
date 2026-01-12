//! LeakyReLU activation function.
//!
//! Applies: leaky_relu(x) = max(0, x) + negative_slope * min(0, x)
//!
//! Improves gradient flow for negative activations compared to standard ReLU.

use bolt_core::backend::{CopyOp, MulOp, NegOp, ReluOp, ReshapeOp, SubOp, SumOp};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

/// LeakyReLU activation layer.
///
/// Uses a small slope for negative values instead of zeroing them completely,
/// which can help with gradient flow during training.
///
/// Default negative_slope is 0.01, which is a common choice in practice.
pub struct LeakyRelu {
    negative_slope: f64,
}

impl Default for LeakyRelu {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl LeakyRelu {
    /// Creates a new LeakyReLU layer with default negative_slope = 0.01.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new LeakyReLU layer with a custom negative slope.
    ///
    /// # Arguments
    ///
    /// * `negative_slope` - The slope for negative values (must be >= 0.0)
    ///
    /// # Panics
    ///
    /// Panics if negative_slope < 0.0
    pub fn with_negative_slope(negative_slope: f64) -> Self {
        assert!(
            negative_slope >= 0.0,
            "negative_slope must be >= 0.0, got {}",
            negative_slope
        );
        Self { negative_slope }
    }
}

impl<B, D> Module<B, D> for LeakyRelu
where
    B: BaseBackend
        + CopyOp<D>
        + ReluOp<D>
        + NegOp<D>
        + MulOp<D>
        + SubOp<D>
        + ReshapeOp<D>
        + SumOp<D>
        + 'static,
    D: Float + PartialOrd + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        Ok(x.leaky_relu(D::from_f64(self.negative_slope))?)
    }
}
