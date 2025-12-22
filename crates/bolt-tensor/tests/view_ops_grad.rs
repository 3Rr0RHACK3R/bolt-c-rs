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
fn reshape_backward_is_identity() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    )?
    .requires_grad();

    let y = x.reshape(&[3, 2])?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_eq!(gx, vec![1.0; 6]);
    Ok(())
}

#[test]
fn transpose_backward_preserves_indexing() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 3],
    )?
    .requires_grad();

    let y = x.transpose(0, 1)?;
    let w = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    )?;
    let loss = y.mul(&w)?.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 1e-6);
    Ok(())
}

#[test]
fn squeeze_unsqueeze_backward_is_identity() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[2, 1, 3],
    )?
    .requires_grad();

    let y = x.squeeze()?;
    let z = y.unsqueeze(1)?;
    let loss = z.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_eq!(gx, vec![1.0; 6]);
    Ok(())
}

#[test]
fn broadcast_to_backward_reduces_to_source_shape() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[3])?
        .requires_grad();
    let y = x.broadcast_to(&[2, 3])?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[2.0, 2.0, 2.0], 1e-6);
    Ok(())
}

#[test]
fn scalar_broadcast_to_backward_accumulates_all_uses() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0_f32], &[])?.requires_grad();
    let y = x.broadcast_to(&[2, 3])?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[6.0], 1e-6);
    Ok(())
}

#[test]
fn expand_backward_sums_replicated_axes() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let x = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0_f32, 2.0, 3.0], &[1, 3])?
        .requires_grad();

    let y = x.expand(&[2, 3])?;
    let loss = y.sum(None, false)?;

    let grads = loss.backward()?;
    let gx = grads.wrt(&x).unwrap().to_vec()?;

    assert_vec_approx_eq(&gx, &[2.0, 2.0, 2.0], 1e-6);
    Ok(())
}
