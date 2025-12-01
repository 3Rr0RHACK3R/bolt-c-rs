use std::sync::Arc;

use bolt_core::Result;
use bolt_core::tensor::Tensor;
use bolt_cpu::CpuBackend;

#[test]
fn neg_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, -2.0, 3.0, -4.0], &[2, 2])?;

    let result = tensor.neg()?.to_vec()?;
    assert_eq!(result, vec![-1.0, 2.0, -3.0, 4.0]);
    Ok(())
}

#[test]
fn neg_f64() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f64>::from_slice(&backend, &[1.0, -2.0, 3.0], &[3])?;

    let result = tensor.neg()?.to_vec()?;
    assert_eq!(result, vec![-1.0, 2.0, -3.0]);
    Ok(())
}

#[test]
fn neg_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[1, -2, 3, -4], &[4])?;

    let result = tensor.neg()?.to_vec()?;
    assert_eq!(result, vec![-1, 2, -3, 4]);
    Ok(())
}

#[test]
fn abs_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, -2.0, -3.5, 4.5], &[4])?;

    let result = tensor.abs()?.to_vec()?;
    assert_eq!(result, vec![1.0, 2.0, 3.5, 4.5]);
    Ok(())
}

#[test]
fn abs_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[1, -2, -3, 4], &[4])?;

    let result = tensor.abs()?.to_vec()?;
    assert_eq!(result, vec![1, 2, 3, 4]);
    Ok(())
}

#[test]
fn exp_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0, 1.0, 2.0], &[3])?;

    let result = tensor.exp()?.to_vec()?;
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 2.718281828).abs() < 1e-6);
    assert!((result[2] - 7.389056099).abs() < 1e-6);
    Ok(())
}

#[test]
fn log_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.718281828, 7.389056099], &[3])?;

    let result = tensor.log()?.to_vec()?;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 2.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn log_negative_produces_nan() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[-1.0], &[1])?;

    let result = tensor.log()?.to_vec()?;
    assert!(result[0].is_nan());
    Ok(())
}

#[test]
fn sqrt_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0, 1.0, 4.0, 9.0], &[4])?;

    let result = tensor.sqrt()?.to_vec()?;
    assert_eq!(result, vec![0.0, 1.0, 2.0, 3.0]);
    Ok(())
}

#[test]
fn sqrt_negative_produces_nan() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[-1.0], &[1])?;

    let result = tensor.sqrt()?.to_vec()?;
    assert!(result[0].is_nan());
    Ok(())
}

#[test]
fn sin_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
        &[3],
    )?;

    let result = tensor.sin()?.to_vec()?;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 1.0).abs() < 1e-6);
    assert!((result[2] - 0.0).abs() < 1e-6);
    Ok(())
}

#[test]
fn cos_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(
        &backend,
        &[0.0, std::f32::consts::PI / 2.0, std::f32::consts::PI],
        &[3],
    )?;

    let result = tensor.cos()?.to_vec()?;
    assert!((result[0] - 1.0).abs() < 1e-6);
    assert!((result[1] - 0.0).abs() < 1e-6);
    assert!((result[2] - (-1.0)).abs() < 1e-6);
    Ok(())
}

#[test]
fn tanh_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[0.0, 1.0, -1.0], &[3])?;

    let result = tensor.tanh()?.to_vec()?;
    assert!((result[0] - 0.0).abs() < 1e-6);
    assert!((result[1] - 0.7615941559).abs() < 1e-6);
    assert!((result[2] - (-0.7615941559)).abs() < 1e-6);
    Ok(())
}

#[test]
fn relu_f32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[-2.0, -1.0, 0.0, 1.0, 2.0], &[5])?;

    let result = tensor.relu()?.to_vec()?;
    assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    Ok(())
}

#[test]
fn relu_i32() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, i32>::from_slice(&backend, &[-2, -1, 0, 1, 2], &[5])?;

    let result = tensor.relu()?.to_vec()?;
    assert_eq!(result, vec![0, 0, 0, 1, 2]);
    Ok(())
}

#[test]
fn unary_ops_on_non_contiguous() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let base =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let permuted = base.permute(&[1, 0])?;
    let result = permuted.neg()?.to_vec()?;
    assert_eq!(result, vec![-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
    Ok(())
}

#[test]
fn chain_unary_ops() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[-1.0, -2.0, 3.0, 4.0], &[4])?;

    let result = tensor.abs()?.sqrt()?.to_vec()?;
    assert_eq!(result, vec![1.0, 2.0f32.sqrt(), 3.0f32.sqrt(), 2.0]);
    Ok(())
}

#[test]
fn exp_large_value_produces_inf() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[1000.0], &[1])?;

    let result = tensor.exp()?.to_vec()?;
    assert!(result[0].is_infinite());
    Ok(())
}

#[test]
fn unary_ops_on_scalar() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor = Tensor::<CpuBackend, f32>::from_slice(&backend, &[2.0], &[1])?;

    let neg_result = tensor.neg()?.to_vec()?;
    assert_eq!(neg_result, vec![-2.0]);

    let sqrt_result = tensor.sqrt()?.to_vec()?;
    assert!((sqrt_result[0] - 1.414213562).abs() < 1e-6);
    Ok(())
}

#[test]
fn unary_ops_preserve_shape() -> Result<()> {
    let backend = Arc::new(CpuBackend::new());
    let tensor =
        Tensor::<CpuBackend, f32>::from_slice(&backend, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let result = tensor.neg()?;
    assert_eq!(result.shape(), &[2, 3]);
    Ok(())
}
