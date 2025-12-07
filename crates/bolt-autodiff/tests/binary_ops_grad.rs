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
fn test_add_grad_simple() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let a = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0], &[3])?.requires_grad();
    let b = Tensor::from_slice(&autodiff, &[4.0_f32, 5.0, 6.0], &[3])?.requires_grad();

    let c = a.add(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &[1.0, 1.0, 1.0], 1e-6);
    assert_vec_approx_eq(&db, &[1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_mul_grad_simple() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let a = Tensor::from_slice(&autodiff, &[2.0_f32, 3.0], &[2])?.requires_grad();
    let b = Tensor::from_slice(&autodiff, &[4.0_f32, 5.0], &[2])?.requires_grad();

    let c = a.mul(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &[4.0, 5.0], 1e-6);
    assert_vec_approx_eq(&db, &[2.0, 3.0], 1e-6);

    Ok(())
}

#[test]
fn test_sub_grad_simple() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let a = Tensor::from_slice(&autodiff, &[5.0_f32, 6.0], &[2])?.requires_grad();
    let b = Tensor::from_slice(&autodiff, &[2.0_f32, 1.0], &[2])?.requires_grad();

    let c = a.sub(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;

    let da = grads.wrt(&a).expect("gradient for a").to_vec()?;
    let db = grads.wrt(&b).expect("gradient for b").to_vec()?;

    assert_vec_approx_eq(&da, &[1.0, 1.0], 1e-6);
    assert_vec_approx_eq(&db, &[-1.0, -1.0], 1e-6);

    Ok(())
}

#[test]
fn test_chain_rule() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[2.0_f32, 3.0], &[2])?.requires_grad();

    let y = x.mul(&x)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[4.0, 6.0], 1e-6);

    Ok(())
}

#[test]
fn test_multiple_use_accumulation() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0], &[2])?.requires_grad();

    let y = x.add(&x)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[2.0, 2.0], 1e-6);

    Ok(())
}

#[test]
fn test_complex_expression() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0], &[2])?.requires_grad();
    let y = Tensor::from_slice(&autodiff, &[3.0_f32, 4.0], &[2])?.requires_grad();

    let a = x.mul(&y)?;
    let b = a.add(&x)?;
    let loss = b.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;
    let dy = grads.wrt(&y).expect("gradient for y").to_vec()?;

    assert_vec_approx_eq(&dx, &[4.0, 5.0], 1e-6);
    assert_vec_approx_eq(&dy, &[1.0, 2.0], 1e-6);

    Ok(())
}

#[test]
fn test_matmul_grad() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let w = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0, 4.0], &[2, 2])?.requires_grad();
    let x = Tensor::from_slice(&autodiff, &[5.0_f32, 6.0, 7.0, 8.0], &[2, 2])?;

    let y = w.matmul(&x)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    let expected = vec![11.0, 15.0, 11.0, 15.0];
    assert_vec_approx_eq(&dw, &expected, 1e-6);

    Ok(())
}
