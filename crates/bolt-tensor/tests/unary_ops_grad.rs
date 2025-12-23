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
fn exp_grad_matches_forward_exp() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0_f32, 1.0], &[2])?.requires_grad();
    let y = x.exp()?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[0.0_f32.exp(), 1.0_f32.exp()], 1e-6);
    Ok(())
}

#[test]
fn log_grad_is_inverse() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 4.0], &[2])?.requires_grad();
    let y = x.log()?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[1.0, 0.25], 1e-6);
    Ok(())
}

#[test]
fn sqrt_grad_is_half_over_sqrt() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[4.0_f32, 9.0], &[2])?.requires_grad();
    let y = x.sqrt()?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[0.25, 1.0 / 6.0], 1e-6);
    Ok(())
}

#[test]
fn tanh_grad_matches_1_minus_tanh_sq() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0_f32, 1.0], &[2])?.requires_grad();
    let y = x.tanh()?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    let t0 = 0.0_f32.tanh();
    let t1 = 1.0_f32.tanh();
    assert_vec_approx_eq(&gx, &[1.0 - t0 * t0, 1.0 - t1 * t1], 1e-6);
    Ok(())
}

#[test]
fn relu_grad_is_zero_for_negative() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[-1.0_f32, 2.0], &[2])?.requires_grad();
    let y = x.relu()?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[0.0, 1.0], 1e-6);
    Ok(())
}

#[test]
fn sigmoid_grad() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0_f32, 1.0], &[2])?.requires_grad();
    let y = x.sigmoid()?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    let sig0 = 1.0 / (1.0 + (-0.0f32).exp());
    let sig1 = 1.0 / (1.0 + (-1.0f32).exp());
    let expected = vec![sig0 * (1.0 - sig0), sig1 * (1.0 - sig1)];
    assert_vec_approx_eq(&gx, &expected, 1e-5);
    Ok(())
}
