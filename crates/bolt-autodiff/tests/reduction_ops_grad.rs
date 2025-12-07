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
fn test_mean_grad() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0, 4.0], &[4])?.requires_grad();

    let loss = x.mean(None, false)?;

    let grads = loss.backward()?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    assert_vec_approx_eq(&dx, &[0.25, 0.25, 0.25, 0.25], 1e-6);

    Ok(())
}

#[test]
fn test_sum_multi_axis_backward() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(
        &autodiff,
        &(1..=24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
    )?
    .requires_grad();

    let y = x.sum(Some(&[0, 2]), false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    let expected_grad = vec![1.0_f32; 24];
    assert_vec_approx_eq(&dx, &expected_grad, 1e-6);

    Ok(())
}

#[test]
fn test_mean_multi_axis_backward() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(
        &autodiff,
        &(1..=24).map(|x| x as f32).collect::<Vec<_>>(),
        &[2, 3, 4],
    )?
    .requires_grad();

    let y = x.mean(Some(&[0, 2]), false)?;

    let grads = y.backward()?;
    let dx = grads.wrt(&x).expect("gradient for x").to_vec()?;

    let expected_grad = vec![0.125_f32; 24];
    assert_vec_approx_eq(&dx, &expected_grad, 1e-6);

    Ok(())
}
