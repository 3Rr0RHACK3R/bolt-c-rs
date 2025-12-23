use std::sync::Arc;

use bolt_core::Result;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

fn assert_vec_approx_eq(actual: &[f32], expected: &[f32], eps: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*a - *e).abs() < eps,
            "idx {i}: got {a}, expected {e} (eps {eps})"
        );
    }
}

#[test]
fn add_backward_reduces_broadcast_vector_over_matrix() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0; 6], &[2, 3])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])?
        .requires_grad();

    let loss = a.add(&b)?.sum(None, false)?;
    let grads = loss.backward()?;

    let ga = grads.wrt(&a).unwrap().to_vec()?;
    let gb = grads.wrt(&b).unwrap().to_vec()?;

    assert_eq!(ga, vec![1.0; 6]);
    assert_eq!(gb, vec![2.0; 3]);
    Ok(())
}

#[test]
fn sub_backward_reduces_broadcast_vector_over_matrix() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0; 6], &[2, 3])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])?
        .requires_grad();

    let loss = a.sub(&b)?.sum(None, false)?;
    let grads = loss.backward()?;

    let ga = grads.wrt(&a).unwrap().to_vec()?;
    let gb = grads.wrt(&b).unwrap().to_vec()?;

    assert_eq!(ga, vec![1.0; 6]);
    assert_eq!(gb, vec![-2.0; 3]);
    Ok(())
}

#[test]
fn mul_backward_reduces_broadcasted_operands() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2, 1])?.requires_grad();
    let b =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0], &[1, 3])?
            .requires_grad();

    let loss = a.mul(&b)?.sum(None, false)?;
    let grads = loss.backward()?;

    let ga = grads.wrt(&a).unwrap().to_vec()?;
    let gb = grads.wrt(&b).unwrap().to_vec()?;

    assert_eq!(ga, vec![60.0, 60.0]);
    assert_eq!(gb, vec![3.0; 3]);
    Ok(())
}

#[test]
fn div_backward_reduces_broadcasted_operands() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 4.0], &[2, 1])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 4.0], &[1, 3])?
        .requires_grad();

    let loss = a.div(&b)?.sum(None, false)?;
    let grads = loss.backward()?;

    let ga = grads.wrt(&a).unwrap().to_vec()?;
    let gb = grads.wrt(&b).unwrap().to_vec()?;

    assert_vec_approx_eq(&ga, &[1.75, 1.75], 1e-6);
    assert_vec_approx_eq(&gb, &[-6.0, -1.5, -0.375], 1e-6);
    Ok(())
}

#[test]
fn add_backward_handles_rank_increase_broadcast() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let a = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])?.requires_grad();
    let b = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0; 6], &[2, 3])?.requires_grad();

    let loss = a.add(&b)?.sum(None, false)?;
    let grads = loss.backward()?;

    let ga = grads.wrt(&a).unwrap().to_vec()?;
    let gb = grads.wrt(&b).unwrap().to_vec()?;

    assert_eq!(ga, vec![2.0; 3]);
    assert_eq!(gb, vec![1.0; 6]);
    Ok(())
}
