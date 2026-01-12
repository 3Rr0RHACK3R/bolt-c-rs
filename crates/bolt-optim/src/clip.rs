use std::collections::HashSet;

use bolt_core::backend::CopyOp;
use bolt_core::{BaseBackend, Error, Float, Result};
use bolt_nn::Param;
use bolt_tensor::{Tensor, no_grad};

/// Epsilon for numerical stability in norm clipping.
const EPS: f64 = 1e-6;

/// Clips gradient norms across all parameters using global L2 norm.
///
/// Computes the global L2 norm of all gradient tensors in `params`, and if
/// it exceeds `max_norm`, scales all gradients by `max_norm / (total_norm + eps)`.
///
/// # Arguments
/// * `params` - Slice of parameters whose gradients will be clipped.
/// * `max_norm` - Maximum allowed L2 norm.
///
/// # Returns
/// The total L2 norm before clipping, or an error if any gradient contains NaN/Inf.
///
/// # Behavior
/// - Skips params with `requires_grad() == false` or `grad() == None`.
/// - Deduplicates params by `Param::key()`.
/// - Returns `Err` on NaN/Inf without modifying any gradients.
/// - If `total_norm == 0` or `total_norm <= max_norm`, gradients are unchanged.
pub fn clip_grad_norm<B, D>(params: &[Param<B, D>], max_norm: D) -> Result<D>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
{
    let max_norm_f64 = max_norm.to_f64();
    if max_norm_f64 <= 0.0 {
        return Err(Error::OpError(format!(
            "clip_grad_norm: max_norm must be positive, got {}",
            max_norm_f64
        )));
    }

    let _ng = no_grad();

    // Phase 1: Collect grads, validate, compute total_norm
    let grad_data = collect_and_validate_grads(params)?;
    if grad_data.is_empty() {
        return Ok(D::zero());
    }

    // Compute global L2 norm
    let sum_sq: f64 = grad_data.iter().map(|(_, _, v)| sum_of_squares(v)).sum();
    let total_norm = sum_sq.sqrt();

    if total_norm == 0.0 || total_norm <= max_norm_f64 {
        return Ok(D::from_f64(total_norm));
    }

    // Compute scale factor
    let scale = max_norm_f64 / (total_norm + EPS);

    // Phase 2: Apply scaling to all grads
    for (param, grad, values) in grad_data {
        let scaled: Vec<D> = values.iter().map(|&v| D::from_f64(v * scale)).collect();
        let new_grad = Tensor::from_vec(&grad.backend(), scaled, grad.shape().as_slice())?;
        param.set_grad(Some(new_grad));
    }

    Ok(D::from_f64(total_norm))
}

/// Clips gradient values to a symmetric range `[-clip_value, +clip_value]`.
///
/// # Arguments
/// * `params` - Slice of parameters whose gradients will be clipped.
/// * `clip_value` - Maximum absolute value for gradient elements.
///
/// # Returns
/// `Ok(())` on success, or an error if any gradient contains NaN/Inf.
///
/// # Behavior
/// - Skips params with `requires_grad() == false` or `grad() == None`.
/// - Deduplicates params by `Param::key()`.
/// - Returns `Err` on NaN/Inf without modifying any gradients.
pub fn clip_grad_value<B, D>(params: &[Param<B, D>], clip_value: D) -> Result<()>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
{
    let clip_f64 = clip_value.to_f64();
    if clip_f64 < 0.0 {
        return Err(Error::OpError(format!(
            "clip_grad_value: clip_value must be non-negative, got {}",
            clip_f64
        )));
    }

    let _ng = no_grad();

    // Phase 1: Collect grads, validate
    let grad_data = collect_and_validate_grads(params)?;
    if grad_data.is_empty() {
        return Ok(());
    }

    // Phase 2: Apply clamping to all grads
    for (param, grad, values) in grad_data {
        let clamped: Vec<D> = values
            .iter()
            .map(|&v| D::from_f64(v.max(-clip_f64).min(clip_f64)))
            .collect();
        let new_grad = Tensor::from_vec(&grad.backend(), clamped, grad.shape().as_slice())?;
        param.set_grad(Some(new_grad));
    }

    Ok(())
}

/// Collects unique trainable params with valid grads, reads values, and validates finiteness.
///
/// Returns a list of (Param, original grad Tensor, grad values as f64).
/// On any NaN/Inf, returns an error immediately (no partial modifications).
fn collect_and_validate_grads<B, D>(
    params: &[Param<B, D>],
) -> Result<Vec<(Param<B, D>, Tensor<B, D>, Vec<f64>)>>
where
    B: BaseBackend + CopyOp<D>,
    D: Float,
{
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for p in params {
        let key = p.key();
        if !seen.insert(key.to_string()) {
            continue;
        }
        if !p.requires_grad() {
            continue;
        }
        let Some(grad) = p.grad() else {
            continue;
        };

        // Validate shape match (mirrors optimizer safety checks)
        if grad.shape() != p.shape() {
            return Err(Error::ShapeMismatch {
                lhs: grad.shape().to_vec(),
                rhs: p.shape().to_vec(),
            });
        }

        // Read grad values to host
        let values_d = grad.to_vec()?;
        let values_f64: Vec<f64> = values_d.iter().map(|&v| v.to_f64()).collect();

        // Validate finiteness
        for (i, &v) in values_f64.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::OpError(format!(
                    "clip_grad: non-finite gradient value {} at index {} in param '{}'",
                    v, i, key
                )));
            }
        }

        result.push((p.clone(), grad, values_f64));
    }

    Ok(result)
}

/// Computes sum of squares for an f64 slice.
#[inline]
fn sum_of_squares(values: &[f64]) -> f64 {
    values.iter().map(|&v| v * v).sum()
}
