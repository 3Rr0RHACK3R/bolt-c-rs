use std::sync::Arc;

use bolt_core::Result;
use bolt_core::error::Error;
use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;

#[test]
fn div_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[4])?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 4.0, 5.0, 8.0], &[4])?;

    let result = lhs.div(&rhs)?.to_vec()?;
    assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
    Ok(())
}

#[test]
fn div_f64() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f64>::from_slice(&backend, &[10.0, 20.0, 30.0], &[3])?;
    let rhs = Tensor::<CpuBackend, f64>::from_slice(&backend, &[2.0, 4.0, 5.0], &[3])?;

    let result = lhs.div(&rhs)?.to_vec()?;
    assert_eq!(result, vec![5.0, 5.0, 6.0]);
    Ok(())
}

#[test]
fn div_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, i32>::from_slice(&backend, &[10, 20, 30], &[3])?;
    let rhs = Tensor::<CpuBackend, i32>::from_slice(&backend, &[2, 4, 5], &[3])?;

    let result = lhs.div(&rhs)?.to_vec()?;
    assert_eq!(result, vec![5, 5, 6]);
    Ok(())
}

#[test]
fn div_by_zero_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0], &[2])?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0, 4.0], &[2])?;

    let result = lhs.div(&rhs)?.to_vec()?;
    assert!(result[0].is_infinite());
    assert_eq!(result[1], 5.0);
    Ok(())
}

#[test]
fn div_by_zero_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, i32>::from_slice(&backend, &[10, 20], &[2])?;
    let rhs = Tensor::<CpuBackend, i32>::from_slice(&backend, &[0, 4], &[2])?;

    let result = lhs.div(&rhs);
    assert!(matches!(result, Err(Error::OpError(_))));
    Ok(())
}

#[test]
fn div_broadcast_scalar() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[4])?;
    let scalar = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?;

    let result = tensor.div(&scalar)?.to_vec()?;
    assert_eq!(result, vec![5.0, 10.0, 15.0, 20.0]);
    Ok(())
}

#[test]
fn div_broadcast_2d() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[3, 2],
    )?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 5.0], &[2])?;

    let result = lhs.div(&rhs)?.to_vec()?;
    assert_eq!(result, vec![5.0, 4.0, 15.0, 8.0, 25.0, 12.0]);
    Ok(())
}

#[test]
fn pow_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0], &[3])?;
    let exp = Tensor::<CpuBackend, f32>::from_slice(&backend, &[3.0, 2.0, 0.5], &[3])?;

    let result = base.pow(&exp)?.to_vec()?;
    assert!((result[0] - 8.0).abs() < 1e-6);
    assert!((result[1] - 9.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn pow_f64() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f64>::from_slice(&backend, &[2.0, 3.0], &[2])?;
    let exp = Tensor::<CpuBackend, f64>::from_slice(&backend, &[3.0, 2.0], &[2])?;

    let result = base.pow(&exp)?.to_vec()?;
    assert!((result[0] - 8.0).abs() < 1e-10);
    assert!((result[1] - 9.0).abs() < 1e-10);
    Ok(())
}

#[test]
fn pow_zero_to_zero() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0], &[1])?;
    let exp = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0], &[1])?;

    let result = base.pow(&exp)?.to_vec()?;
    assert_eq!(result[0], 1.0);
    Ok(())
}

#[test]
fn pow_negative_base_fractional_exp() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &[-1.0], &[1])?;
    let exp = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.5], &[1])?;

    let result = base.pow(&exp)?.to_vec()?;
    assert!(result[0].is_nan());
    Ok(())
}

#[test]
fn pow_broadcast_scalar() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0], &[3])?;
    let exp = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?;

    let result = base.pow(&exp)?.to_vec()?;
    assert_eq!(result, vec![4.0, 9.0, 16.0]);
    Ok(())
}

#[test]
fn pow_broadcast_2d() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0, 5.0], &[2, 2])?;
    let exp = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0], &[2])?;

    let result = base.pow(&exp)?.to_vec()?;
    assert_eq!(result, vec![4.0, 27.0, 16.0, 125.0]);
    Ok(())
}

#[test]
fn div_i32_min_overflow() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, i32>::from_slice(&backend, &[i32::MIN, 10], &[2])?;
    let rhs = Tensor::<CpuBackend, i32>::from_slice(&backend, &[-1, 2], &[2])?;

    let result = lhs.div(&rhs);
    assert!(matches!(result, Err(Error::OpError(_))));
    Ok(())
}

#[test]
fn add_broadcast_2d() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0], &[2])?;

    let result = lhs.add(&rhs)?.to_vec()?;
    assert_eq!(result, vec![11.0, 22.0, 13.0, 24.0]);
    Ok(())
}

#[test]
fn sub_broadcast_2d() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 20.0, 30.0, 40.0], &[2, 2])?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0], &[2])?;

    let result = lhs.sub(&rhs)?.to_vec()?;
    assert_eq!(result, vec![9.0, 18.0, 29.0, 38.0]);
    Ok(())
}

#[test]
fn mul_broadcast_2d() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let lhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0, 3.0, 4.0, 5.0], &[2, 2])?;
    let rhs = Tensor::<CpuBackend, f32>::from_slice(&backend, &[10.0, 100.0], &[2])?;

    let result = lhs.mul(&rhs)?.to_vec()?;
    assert_eq!(result, vec![20.0, 300.0, 40.0, 500.0]);
    Ok(())
}
