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
fn test_backward_on_intermediate_tensor() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let w = Tensor::from_slice(&autodiff, &[2.0_f32, 3.0], &[2])?.requires_grad();
    let y = w.mul(&w)?;

    let _z = y.mul(&w)?;

    let grads = y.backward()?;

    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[4.0, 6.0], 1e-6);

    Ok(())
}

#[test]
fn test_backward_on_non_scalar_tensor() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let w = Tensor::from_slice(&autodiff, &[2.0_f32, 3.0], &[2])?.requires_grad();
    let c = Tensor::from_slice(&autodiff, &[10.0_f32, 100.0], &[2])?;

    let y = w.mul(&c)?;

    let grads = y.backward()?;

    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[10.0, 100.0], 1e-6);

    Ok(())
}
