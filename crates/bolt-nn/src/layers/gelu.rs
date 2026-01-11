//! GELU (Gaussian Error Linear Unit) activation function.
//!
//! Uses the tanh approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//! Reference: https://arxiv.org/abs/1606.08415

use bolt_core::backend::{
    AddOp, CopyOp, FillOp, LogOp, MulOp, NegOp, PowOp, ReshapeOp, SubOp, SumOp, TanhOp,
};
use bolt_core::{BaseBackend, Float};
use bolt_tensor::Tensor;

use crate::{ForwardCtx, Module, Result};

/// GELU activation using tanh approximation.
///
/// Common in transformer architectures. The formula used is:
/// `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))`
#[derive(Default)]
pub struct Gelu;

impl Gelu {
    pub fn new() -> Self {
        Self
    }
}

/// sqrt(2 / PI) ≈ 0.7978845608
const SQRT_2_OVER_PI: f64 = 0.7978845608028654;

/// Coefficient for x^3 term in tanh approximation
const COEFF: f64 = 0.044715;

impl<B, D> Module<B, D> for Gelu
where
    B: BaseBackend
        + CopyOp<D>
        + FillOp<D>
        + AddOp<D>
        + MulOp<D>
        + PowOp<D>
        + TanhOp<D>
        + SubOp<D>
        + NegOp<D>
        + LogOp<D>
        + ReshapeOp<D>
        + SumOp<D>
        + 'static,
    D: Float + 'static,
{
    fn forward(&self, x: Tensor<B, D>, _ctx: &mut ForwardCtx) -> Result<Tensor<B, D>> {
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

        let backend = x.backend();

        let half = Tensor::from_slice(&backend, &[D::from_f64(0.5)], &[])?;
        let one = Tensor::from_slice(&backend, &[D::from_f64(1.0)], &[])?;
        let three = Tensor::from_slice(&backend, &[D::from_f64(3.0)], &[])?;
        let sqrt_2_pi = Tensor::from_slice(&backend, &[D::from_f64(SQRT_2_OVER_PI)], &[])?;
        let coeff = Tensor::from_slice(&backend, &[D::from_f64(COEFF)], &[])?;

        let x_cubed = x.pow(&three)?;
        let coeff_x_cubed = x_cubed.mul(&coeff)?;
        let inner = x.add(&coeff_x_cubed)?;
        let tanh_input = inner.mul(&sqrt_2_pi)?;
        let tanh_result = tanh_input.tanh()?;
        let one_plus_tanh = one.add(&tanh_result)?;
        let x_times_bracket = x.mul(&one_plus_tanh)?;
        let result = half.mul(&x_times_bracket)?;

        Ok(result)
    }
}
