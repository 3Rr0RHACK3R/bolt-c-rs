use std::sync::Arc;

use bolt_core::Result;
use bolt_core::error::Error;
use bolt_cpu::CpuBackend;
use bolt_tensor::Tensor;

#[test]
fn broadcast_scalar_to_vector() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let scalar = Tensor::<CpuBackend, f32>::from_slice(&backend, &[5.0], &[1])?;
    let broadcasted = scalar.broadcast_to(&[3])?;

    assert_eq!(broadcasted.shape(), &[3]);
    assert_eq!(broadcasted.to_vec()?, vec![5.0, 5.0, 5.0]);

    Ok(())
}

#[test]
fn broadcast_rank_increase_updates_strides() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let vector = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?;
    let broadcasted = vector.broadcast_to(&[3, 2])?;

    assert_eq!(broadcasted.shape(), &[3, 2]);
    let strides = broadcasted.strides();
    assert_eq!(strides[0], 0);
    assert_ne!(strides[1], 0);

    let expected = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0];
    assert_eq!(broadcasted.to_vec()?, expected);

    Ok(())
}

#[test]
fn broadcast_incompatible_shape_errors() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0], &[3])?;
    let err = tensor.broadcast_to(&[2]);

    assert!(matches!(err, Err(Error::ShapeMismatch { .. })));
    Ok(())
}

#[test]
fn broadcast_then_elementwise_ops() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let scalar = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0], &[1])?;
    let values = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0], &[3])?;

    let broadcasted = scalar.broadcast_to(&[3])?;
    let sum = broadcasted.add(&values)?.to_vec()?;
    let prod = broadcasted.mul(&values)?.to_vec()?;

    assert_eq!(sum, vec![3.0, 4.0, 5.0]);
    assert_eq!(prod, vec![2.0, 3.0, 4.0]);

    Ok(())
}

#[test]
fn broadcast_then_matmul() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());

    let row = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 0.0], &[1, 2])?;
    let weights = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2, 1])?;

    let broadcasted = row.broadcast_to(&[3, 2])?;
    let product = broadcasted.matmul(&weights)?.to_vec()?;

    assert_eq!(product, vec![2.0, 2.0, 2.0]);

    Ok(())
}
