use std::sync::Arc;

use bolt_cpu::CpuBackend;
use bolt_nn::{Init, Store};
use bolt_optim::{clip_grad_norm, clip_grad_value};
use bolt_tensor::Tensor;

type B = CpuBackend;
type D = f32;

fn backend() -> Arc<B> {
    Arc::new(CpuBackend::new())
}

// ============================================================================
// clip_grad_norm tests
// ============================================================================

#[test]
fn clip_grad_norm_scales_global_l2() {
    // Given two params with grads [3, 4] and [0], norm = sqrt(9+16+0) = 5.
    // clip_grad_norm(max_norm=2) should return 5 and scale grads by 2/5.
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p1 = store.param("p1", &[2], Init::Zeros).unwrap();
    let p2 = store.param("p2", &[1], Init::Zeros).unwrap();

    // Set grads
    p1.set_grad(Some(Tensor::from_slice(&b, &[3.0, 4.0], &[2]).unwrap()));
    p2.set_grad(Some(Tensor::from_slice(&b, &[0.0], &[1]).unwrap()));

    let total_norm = clip_grad_norm(&[p1.clone(), p2.clone()], 2.0).unwrap();

    assert!((total_norm - 5.0).abs() < 1e-5, "expected norm ~5.0, got {}", total_norm);

    // Check scaled grads: scale = 2 / (5 + 1e-6) ≈ 0.4
    let scale = 2.0 / (5.0 + 1e-6);
    let g1 = p1.grad().unwrap().to_vec().unwrap();
    let g2 = p2.grad().unwrap().to_vec().unwrap();

    assert!((g1[0] - 3.0 * scale as f32).abs() < 1e-5);
    assert!((g1[1] - 4.0 * scale as f32).abs() < 1e-5);
    assert!((g2[0] - 0.0).abs() < 1e-5);
}

#[test]
fn clip_grad_norm_noop_when_below_threshold() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[3.0, 4.0], &[2]).unwrap()));

    // norm = 5, max_norm = 10 => no scaling
    let total_norm = clip_grad_norm(&[p.clone()], 10.0).unwrap();

    assert!((total_norm - 5.0).abs() < 1e-5);

    let g = p.grad().unwrap().to_vec().unwrap();
    assert!((g[0] - 3.0).abs() < 1e-5);
    assert!((g[1] - 4.0).abs() < 1e-5);
}

#[test]
fn clip_grad_norm_noop_when_zero() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[0.0, 0.0], &[2]).unwrap()));

    let total_norm = clip_grad_norm(&[p.clone()], 1.0).unwrap();

    assert!((total_norm - 0.0).abs() < 1e-5);

    let g = p.grad().unwrap().to_vec().unwrap();
    assert!((g[0] - 0.0).abs() < 1e-5);
    assert!((g[1] - 0.0).abs() < 1e-5);
}

#[test]
fn clip_grad_norm_errors_on_nan_and_does_not_modify() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p1 = store.param("p1", &[2], Init::Zeros).unwrap();
    let p2 = store.param("p2", &[1], Init::Zeros).unwrap();

    // p1 has valid grad, p2 has NaN
    p1.set_grad(Some(Tensor::from_slice(&b, &[1.0, 2.0], &[2]).unwrap()));
    p2.set_grad(Some(Tensor::from_slice(&b, &[f32::NAN], &[1]).unwrap()));

    let result = clip_grad_norm(&[p1.clone(), p2.clone()], 1.0);
    assert!(result.is_err());

    // Verify p1's grad is unchanged (no partial modification)
    let g1 = p1.grad().unwrap().to_vec().unwrap();
    assert!((g1[0] - 1.0).abs() < 1e-5);
    assert!((g1[1] - 2.0).abs() < 1e-5);
}

#[test]
fn clip_grad_norm_errors_on_inf() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[1], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[f32::INFINITY], &[1]).unwrap()));

    let result = clip_grad_norm(&[p.clone()], 1.0);
    assert!(result.is_err());
}

#[test]
fn clip_grad_norm_errors_on_invalid_max_norm() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);
    let p = store.param("p", &[1], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[1.0], &[1]).unwrap()));

    // max_norm <= 0 should error
    assert!(clip_grad_norm(&[p.clone()], 0.0).is_err());
    assert!(clip_grad_norm(&[p.clone()], -1.0).is_err());
}

#[test]
fn clip_grad_norm_empty_params() {
    let params: &[bolt_nn::Param<B, D>] = &[];
    let result = clip_grad_norm(params, 1.0).unwrap();
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn clip_grad_norm_all_grads_none() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    // Don't set grad - it remains None

    let result = clip_grad_norm(&[p.clone()], 1.0).unwrap();
    assert!((result - 0.0).abs() < 1e-5);
}

#[test]
fn clip_grad_norm_skips_frozen_params() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p1 = store.param("p1", &[2], Init::Zeros).unwrap();
    let p2 = store.param("p2", &[1], Init::Zeros).unwrap();

    // Freeze p1
    p1.freeze();
    p1.set_grad(Some(Tensor::from_slice(&b, &[100.0, 100.0], &[2]).unwrap()));
    p2.set_grad(Some(Tensor::from_slice(&b, &[3.0], &[1]).unwrap()));

    // Only p2's grad should contribute to norm
    let total_norm = clip_grad_norm(&[p1.clone(), p2.clone()], 10.0).unwrap();
    assert!((total_norm - 3.0).abs() < 1e-5);
}

// ============================================================================
// clip_grad_value tests
// ============================================================================

#[test]
fn clip_grad_value_clamps_symmetric() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[4], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[-5.0, -0.5, 0.5, 5.0], &[4]).unwrap()));

    clip_grad_value(&[p.clone()], 1.0).unwrap();

    let g = p.grad().unwrap().to_vec().unwrap();
    assert!((g[0] - (-1.0)).abs() < 1e-5); // -5 clamped to -1
    assert!((g[1] - (-0.5)).abs() < 1e-5); // -0.5 unchanged
    assert!((g[2] - 0.5).abs() < 1e-5);    // 0.5 unchanged
    assert!((g[3] - 1.0).abs() < 1e-5);    // 5 clamped to 1
}

#[test]
fn clip_grad_value_errors_on_inf_and_does_not_modify() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p1 = store.param("p1", &[2], Init::Zeros).unwrap();
    let p2 = store.param("p2", &[1], Init::Zeros).unwrap();

    p1.set_grad(Some(Tensor::from_slice(&b, &[1.0, 2.0], &[2]).unwrap()));
    p2.set_grad(Some(Tensor::from_slice(&b, &[f32::NEG_INFINITY], &[1]).unwrap()));

    let result = clip_grad_value(&[p1.clone(), p2.clone()], 1.0);
    assert!(result.is_err());

    // p1's grad should be unchanged
    let g1 = p1.grad().unwrap().to_vec().unwrap();
    assert!((g1[0] - 1.0).abs() < 1e-5);
    assert!((g1[1] - 2.0).abs() < 1e-5);
}

#[test]
fn clip_grad_value_errors_on_invalid_clip_value() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);
    let p = store.param("p", &[1], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[1.0], &[1]).unwrap()));

    // clip_value < 0 should error
    assert!(clip_grad_value(&[p.clone()], -1.0).is_err());
}

#[test]
fn clip_grad_value_zero_clip_value_zeros_grads() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[5.0, -3.0], &[2]).unwrap()));

    clip_grad_value(&[p.clone()], 0.0).unwrap();

    let g = p.grad().unwrap().to_vec().unwrap();
    assert!((g[0] - 0.0).abs() < 1e-5);
    assert!((g[1] - 0.0).abs() < 1e-5);
}

#[test]
fn clip_grad_value_empty_params() {
    let params: &[bolt_nn::Param<B, D>] = &[];
    let result = clip_grad_value(params, 1.0);
    assert!(result.is_ok());
}

// ============================================================================
// Deduplication tests
// ============================================================================

#[test]
fn dedupes_duplicate_params_by_key_for_norm() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[3.0, 4.0], &[2]).unwrap()));

    // Pass the same param multiple times
    let total_norm = clip_grad_norm(&[p.clone(), p.clone(), p.clone()], 2.0).unwrap();

    // If deduplication works, norm should be 5 (not some multiple)
    assert!((total_norm - 5.0).abs() < 1e-5);
}

#[test]
fn dedupes_duplicate_params_by_key_for_value() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[5.0, -5.0], &[2]).unwrap()));

    // Pass the same param multiple times - should still work correctly
    clip_grad_value(&[p.clone(), p.clone()], 1.0).unwrap();

    let g = p.grad().unwrap().to_vec().unwrap();
    assert!((g[0] - 1.0).abs() < 1e-5);
    assert!((g[1] - (-1.0)).abs() < 1e-5);
}

// ============================================================================
// Shape mismatch tests
// ============================================================================

#[test]
fn clip_grad_norm_errors_on_shape_mismatch() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    // Set a grad with wrong shape
    p.set_grad(Some(Tensor::from_slice(&b, &[1.0, 2.0, 3.0], &[3]).unwrap()));

    let result = clip_grad_norm(&[p.clone()], 1.0);
    assert!(result.is_err());
}

#[test]
fn clip_grad_value_errors_on_shape_mismatch() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p = store.param("p", &[2], Init::Zeros).unwrap();
    p.set_grad(Some(Tensor::from_slice(&b, &[1.0], &[1]).unwrap()));

    let result = clip_grad_value(&[p.clone()], 1.0);
    assert!(result.is_err());
}

// ============================================================================
// Integration-style tests
// ============================================================================

#[test]
fn clip_grad_norm_with_multiple_params_various_shapes() {
    let b = backend();
    let store = Store::<B, D>::new(b.clone(), 42);

    let p1 = store.param("w1", &[2, 3], Init::Zeros).unwrap();
    let p2 = store.param("w2", &[4], Init::Zeros).unwrap();
    let p3 = store.param("b1", &[3], Init::Zeros).unwrap();

    // Set grads: all ones
    p1.set_grad(Some(Tensor::from_slice(&b, &[1.0; 6], &[2, 3]).unwrap()));
    p2.set_grad(Some(Tensor::from_slice(&b, &[1.0; 4], &[4]).unwrap()));
    p3.set_grad(Some(Tensor::from_slice(&b, &[1.0; 3], &[3]).unwrap()));

    // Total elements = 6 + 4 + 3 = 13, all ones => norm = sqrt(13)
    let expected_norm = (13.0_f64).sqrt();
    let total_norm = clip_grad_norm(&[p1.clone(), p2.clone(), p3.clone()], 1.0).unwrap();

    assert!((total_norm as f64 - expected_norm).abs() < 1e-4);

    // After clipping to max_norm=1, each grad element should be scaled by 1/sqrt(13)
    let scale = 1.0 / (expected_norm + 1e-6);
    let g1 = p1.grad().unwrap().to_vec().unwrap();
    for v in g1 {
        assert!((v as f64 - scale).abs() < 1e-4);
    }
}
