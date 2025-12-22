use std::sync::Arc;

use bolt_core::Result;
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
fn add_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])?.requires_grad();
    let b =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0, 5.0, 6.0], &[3])?.requires_grad();

    let c = a.add(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;
    let da = grads.wrt(&a).unwrap().to_vec()?;
    let db = grads.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&da, &[1.0, 1.0, 1.0], 1e-6);
    assert_vec_approx_eq(&db, &[1.0, 1.0, 1.0], 1e-6);
    Ok(())
}

#[test]
fn mul_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0, 5.0], &[2])?.requires_grad();

    let c = a.mul(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;
    let da = grads.wrt(&a).unwrap().to_vec()?;
    let db = grads.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&da, &[4.0, 5.0], 1e-6);
    assert_vec_approx_eq(&db, &[2.0, 3.0], 1e-6);
    Ok(())
}

#[test]
fn sub_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[5.0, 6.0], &[2])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 1.0], &[2])?.requires_grad();

    let c = a.sub(&b)?;
    let loss = c.sum(None, false)?;

    let grads = loss.backward()?;
    let da = grads.wrt(&a).unwrap().to_vec()?;
    let db = grads.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&da, &[1.0, 1.0], 1e-6);
    assert_vec_approx_eq(&db, &[-1.0, -1.0], 1e-6);
    Ok(())
}

#[test]
fn chain_rule_square() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?.requires_grad();
    let y = x.mul(&x)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[4.0, 6.0], 1e-6);
    Ok(())
}

#[test]
fn multiple_use_accumulates_grads() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?.requires_grad();
    let y = x.add(&x)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[2.0, 2.0], 1e-6);
    Ok(())
}

#[test]
fn complex_expression_grads_match_expected() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?.requires_grad();
    let y = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 4.0], &[2])?.requires_grad();

    let a = x.mul(&y)?;
    let b = a.add(&x)?;
    let loss = b.sum(None, false)?;

    let grads = loss.backward()?;
    let dx = grads.wrt(&x).unwrap().to_vec()?;
    let dy = grads.wrt(&y).unwrap().to_vec()?;

    assert_vec_approx_eq(&dx, &[4.0, 5.0], 1e-6);
    assert_vec_approx_eq(&dy, &[1.0, 2.0], 1e-6);
    Ok(())
}

#[test]
fn matmul_grad_matches_expected() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let w = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?
        .requires_grad();
    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[5.0, 6.0, 7.0, 8.0], &[2, 2])?;

    let y = w.matmul(&x)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let dw = grads.wrt(&w).unwrap().to_vec()?;

    assert_vec_approx_eq(&dw, &[11.0, 15.0, 11.0, 15.0], 1e-6);
    Ok(())
}

#[test]
fn div_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[6.0, 8.0], &[2])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 4.0], &[2])?.requires_grad();

    let y = a.div(&b)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let da = grads.wrt(&a).unwrap().to_vec()?;
    let db = grads.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&da, &[0.5, 0.25], 1e-6);
    assert_vec_approx_eq(&db, &[-1.5, -0.5], 1e-6);
    Ok(())
}

#[test]
fn pow_grad_simple() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 2.0], &[2])?.requires_grad();

    let y = a.pow(&b)?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let da = grads.wrt(&a).unwrap().to_vec()?;
    let db = grads.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&da, &[12.0, 6.0], 1e-5);
    assert_vec_approx_eq(&db, &[8.0 * 2.0_f32.ln(), 9.0 * 3.0_f32.ln()], 1e-5);
    Ok(())
}
