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
fn backward_on_intermediate_tensor_accumulates_correctly() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let w = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?.requires_grad();
    let y = w.mul(&w)?;
    let _z = y.mul(&w)?;

    let grads = y.backward()?;
    let dw = grads.wrt(&w).unwrap().to_vec()?;
    assert_vec_approx_eq(&dw, &[4.0, 6.0], 1e-6);
    Ok(())
}

#[test]
fn backward_on_non_scalar_tensor_seeds_ones_like() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let w = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?.requires_grad();
    let c = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 100.0], &[2])?;
    let y = w.mul(&c)?;

    let grads = y.backward()?;
    let dw = grads.wrt(&w).unwrap().to_vec()?;
    assert_vec_approx_eq(&dw, &[10.0, 100.0], 1e-6);
    Ok(())
}

#[test]
fn backward_is_fresh_per_call() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0_f32, 3.0], &[2])?.requires_grad();

    let loss1 = x.mul(&x)?.sum(None, false)?;
    let g1 = loss1.backward()?;
    let dx1 = g1.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx1, &[4.0, 6.0], 1e-6);

    let loss2 = x.mul(&x)?.sum(None, false)?;
    let g2 = loss2.backward()?;
    let dx2 = g2.wrt(&x).unwrap().to_vec()?;
    assert_vec_approx_eq(&dx2, &[4.0, 6.0], 1e-6);

    Ok(())
}
