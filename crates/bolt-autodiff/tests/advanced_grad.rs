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
fn test_dynamic_control_flow_branch_true() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[3.0_f32], &[1])?.requires_grad();
    let w1 = Tensor::from_slice(&autodiff, &[2.0_f32], &[1])?.requires_grad();
    let w2 = Tensor::from_slice(&autodiff, &[5.0_f32], &[1])?.requires_grad();

    let condition_value = x.item()?;

    let loss = if condition_value > 2.0 {
        x.mul(&w1)?.sum(None, false)?
    } else {
        x.mul(&w2)?.sum(None, false)?
    };

    let grads = loss.backward()?;

    let dw1 = grads.wrt(&w1).expect("gradient for w1").to_vec()?;
    assert_vec_approx_eq(&dw1, &[3.0], 1e-6);

    assert!(grads.wrt(&w2).is_none());

    Ok(())
}

#[test]
fn test_dynamic_control_flow_branch_false() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32], &[1])?.requires_grad();
    let w1 = Tensor::from_slice(&autodiff, &[2.0_f32], &[1])?.requires_grad();
    let w2 = Tensor::from_slice(&autodiff, &[5.0_f32], &[1])?.requires_grad();

    let condition_value = x.item()?;

    let loss = if condition_value > 2.0 {
        x.mul(&w1)?.sum(None, false)?
    } else {
        x.mul(&w2)?.sum(None, false)?
    };

    let grads = loss.backward()?;

    let dw2 = grads.wrt(&w2).expect("gradient for w2").to_vec()?;
    assert_vec_approx_eq(&dw2, &[1.0], 1e-6);

    assert!(grads.wrt(&w1).is_none());

    Ok(())
}

#[test]
fn test_dynamic_loop_iterations() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[2.0_f32], &[1])?.requires_grad();
    let w = Tensor::from_slice(&autodiff, &[3.0_f32], &[1])?.requires_grad();

    let iterations = x.item()? as usize;

    let mut result = w.mul(&w)?;
    for _ in 1..iterations {
        result = result.mul(&w)?;
    }
    let loss = result.sum(None, false)?;

    let grads = loss.backward()?;

    let dw = grads.wrt(&w).expect("gradient for w").to_vec()?;
    assert_vec_approx_eq(&dw, &[27.0], 1e-6);

    Ok(())
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
