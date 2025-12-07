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
fn test_reshape_gradients() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?
        .requires_grad();

    let y = x.reshape(&[6])?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx.len(), 6);
    assert_vec_approx_eq(&dx, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_transpose_gradients() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?
        .requires_grad();

    let y = x.transpose(0, 1)?;
    assert_eq!(y.shape(), &[3, 2]);

    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx.len(), 6);
    assert_vec_approx_eq(&dx, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_squeeze_unsqueeze_gradients() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?
        .requires_grad();

    let y = x.unsqueeze(1)?;
    assert_eq!(y.shape(), &[2, 1, 3]);

    let z = y.squeeze_axis(1)?;
    assert_eq!(z.shape(), &[2, 3]);

    let loss = z.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx.len(), 6);
    assert_vec_approx_eq(&dx, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1e-6);

    Ok(())
}

#[test]
fn test_expand_gradients() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0], &[1, 3])?.requires_grad();

    let y = x.expand(&[2, 3])?;
    assert_eq!(y.shape(), &[2, 3]);

    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx.len(), 3);
    assert_vec_approx_eq(&dx, &[2.0, 2.0, 2.0], 1e-6);

    Ok(())
}

#[test]
fn test_broadcast_to_gradients() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0], &[3])?.requires_grad();

    let y = x.broadcast_to(&[2, 3])?;
    assert_eq!(y.shape(), &[2, 3]);

    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx.len(), 3);
    assert_vec_approx_eq(&dx, &[2.0, 2.0, 2.0], 1e-6);

    Ok(())
}

#[test]
fn test_combined_view_ops_gradients() -> Result<()> {
    let cpu_backend = Arc::new(CpuBackend::new());
    let autodiff = Arc::new(Autodiff::wrap(cpu_backend.clone()));

    let _ctx = autodiff.begin_grad();

    let x = Tensor::from_slice(&autodiff, &[1.0_f32, 2.0, 3.0, 4.0], &[4])?.requires_grad();

    let y = x.reshape(&[2, 2])?;
    let z = y.unsqueeze(0)?;
    let w = z.transpose(1, 2)?;
    let v = w.squeeze_axis(0)?;

    let loss = v.sum(None, false)?;

    let grads = loss.backward()?;

    let dx = grads.wrt(&x).unwrap().to_vec()?;
    assert_eq!(dx.len(), 4);
    assert_vec_approx_eq(&dx, &[1.0, 1.0, 1.0, 1.0], 1e-6);

    Ok(())
}
