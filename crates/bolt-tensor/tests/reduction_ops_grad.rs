use std::sync::Arc;

use bolt_core::{Result, error::Error};
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(approx_eq(*a, *e, eps), "idx {i}: got {a}, expected {e}");
    }
}

#[test]
fn mean_grad_is_uniform() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[4])?
        .requires_grad();

    let loss = x.mean(None, false)?;
    let grads = loss.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.25, 0.25, 0.25, 0.25], 1e-6);
    Ok(())
}

#[test]
fn sum_multi_axis_backward_is_ones() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?.requires_grad();

    let y = x.sum(Some(&[0, 2]), false)?;
    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[1.0; 24], 1e-6);
    Ok(())
}

#[test]
fn mean_multi_axis_backward_scales_by_num_reduced() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &data, &[2, 3, 4])?.requires_grad();

    let y = x.mean(Some(&[0, 2]), false)?;
    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.125; 24], 1e-6);
    Ok(())
}

#[test]
fn min_grad_splits_on_ties() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 1.0, 4.0, 1.0], &[4])?
        .requires_grad();
    let y = x.min(None, false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.0, 0.5, 0.0, 0.5], 1e-6);
    Ok(())
}

#[test]
fn max_grad_splits_on_ties() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 3.0, 2.0, 3.0], &[4])?
        .requires_grad();
    let y = x.max(None, false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.0, 0.5, 0.0, 0.5], 1e-6);
    Ok(())
}

#[test]
fn prod_grad_handles_zeros() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 0.0, 4.0], &[3])?.requires_grad();
    let y = x.prod(None, false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.0, 8.0, 0.0], 1e-6);
    Ok(())
}

#[test]
fn prod_grad_matches_expected() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0], &[3])?.requires_grad();
    let y = x.prod(None, false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[12.0, 8.0, 6.0], 1e-6);
    Ok(())
}

#[test]
fn prod_grad_multiple_zeros_is_zero() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0, 0.0, 3.0], &[3])?.requires_grad();
    let y = x.prod(None, false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.0, 0.0, 0.0], 1e-6);
    Ok(())
}

#[test]
fn min_grad_multi_axis_matches_expected() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 0.5, 4.0, 5.0], &[2, 3])?
            .requires_grad();

    let y = x.min(Some(&[0]), false)?;
    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0], 1e-6);
    Ok(())
}

#[test]
fn max_grad_keepdims_multi_axis_splits_on_ties() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 5.0, 3.0, 5.0], &[2, 2])?
        .requires_grad();

    let y = x.max(Some(&[0]), true)?;
    let grads = y.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[0.0, 0.5, 1.0, 0.5], 1e-6);
    Ok(())
}

#[test]
fn argmin_requires_grad_errors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?.requires_grad();

    let err = x.argmin(None, false).unwrap_err();
    match err {
        Error::OpError(msg) => assert!(msg.contains("argmin")),
        other => panic!("unexpected error: {other:?}"),
    }
    Ok(())
}

#[test]
fn argmax_requires_grad_errors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?.requires_grad();

    let err = x.argmax(None, false).unwrap_err();
    match err {
        Error::OpError(msg) => assert!(msg.contains("argmax")),
        other => panic!("unexpected error: {other:?}"),
    }
    Ok(())
}
