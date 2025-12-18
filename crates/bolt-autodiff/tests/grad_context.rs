use bolt_autodiff::{Autodiff, AutodiffTensorExt, Result};
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;
use std::sync::Arc;

fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() < eps
}

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "vector lengths differ: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            approx_eq(*a, *e, eps),
            "element {} differs: got {}, expected {} (eps={})",
            i,
            a,
            e,
            eps
        );
    }
}

#[test]
fn test_no_grad_tensor() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0], &[2])?;
    let y = Tensor::from_slice(&autodiff, &[3.0_f32, 4.0], &[2])?.requires_grad();

    let z = x.add(&y)?;
    let loss = z.sum(None, false)?;

    let grads = loss.backward()?;

    assert!(grads.wrt(&x).is_none());
    assert!(grads.wrt(&y).is_some());

    let dy = grads.wrt(&y).expect("gradient for y").to_vec()?;
    assert_vec_approx_eq(&dy, &[1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_detach() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[2.0_f32, 3.0], &[2])?.requires_grad();
    let w = Tensor::from_slice(&autodiff, &[1.0_f32, 1.0], &[2])?.requires_grad();

    let y = x.mul(&x)?;
    let y_detached = y.detach();
    let y_autodiff = Tensor::from_slice(&autodiff, &y_detached.to_vec()?, &[2])?;
    let z = y_autodiff.add(&w)?;
    let loss = z.sum(None, false)?;

    let grads = loss.backward()?;

    assert!(grads.wrt(&x).is_none());
    assert!(grads.wrt(&w).is_some());

    Ok(())
}

#[test]
fn test_no_grad_guard() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0], &[2])?.requires_grad();

    let y = {
        let _guard = autodiff.no_grad();
        x.add(&x)?
    };

    assert!(!y.is_tracked());

    Ok(())
}

#[test]
fn test_requires_grad_propagation() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let a = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0], &[3])?.requires_grad();
    let b = Tensor::from_slice(&autodiff, &[4.0_f32, 5.0, 6.0], &[3])?;

    assert!(a.is_tracked());
    assert!(!b.is_tracked());

    let c = a.add(&b)?;
    assert!(c.is_tracked());

    Ok(())
}

#[test]
fn test_backward_delegation_cross_graph() -> Result<()> {
    let cpu = Arc::new(CpuBackend::new());
    let ad1 = Arc::new(Autodiff::wrap(cpu.clone()));
    let ad2 = Arc::new(Autodiff::wrap(cpu.clone()));

    let _ctx1 = ad1.begin_grad();
    let _ctx2 = ad2.begin_grad();

    let x = Tensor::from_slice(&ad1, &[2.0_f32], &[1])?.requires_grad();
    let y = x.mul(&x)?; // y is in ad1's graph

    // Even if we use ctx2, it should work because it delegates to y.backward(),
    // which uses y's backend (ad1).
    let ad2_ctx = ad2.begin_grad();
    let grads = ad2_ctx.backward(&y)?;

    assert!(grads.wrt(&x).is_some());
    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx[0], 4.0); // d(x^2)/dx = 2x = 4.0

    Ok(())
}
